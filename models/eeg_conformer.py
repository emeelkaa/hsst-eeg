import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40, n_channels: int = 18):
        super().__init__()
        self.emb_size = emb_size
        self.n_channels = n_channels

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)), 
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, drop_p: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(drop_p)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = torch.nn.functional.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int, num_heads: int, drop_p: float = 0.5, forward_expansion: int = 4, forward_drop_p: float = 0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int, num_heads: int):
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class EEGConformer(nn.Module):
    def __init__(self, emb_size: int = 40, depth: int = 6, num_heads: int = 4, n_channels: int = 18, n_classes: int = 1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.encoder = TransformerEncoder(depth, emb_size, num_heads)
        self.clshead = ClassificationHead(emb_size, n_classes)
    
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.patch_embedding(x)
        x = self.encoder(x) 
        x = self.clshead(x)
        return x
    

if __name__ == '__main__':
    import time
    model = EEGConformer(emb_size=64, depth=2, num_heads=4, n_channels=18, n_classes=1).to('cuda')
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
    sample = torch.randn(1, 18, 2560).to('cuda')
    t0 = time.time()
    out = model(sample)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(f"Output shape: {out.shape}")
    