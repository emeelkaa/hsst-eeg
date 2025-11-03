import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from mamba_ssm import Mamba


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40, num_channels: int = 16):
        super().__init__()
        self.emb_size = emb_size

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, 25), (1, 1)),
            nn.Conv2d(emb_size, emb_size, (num_channels, 1), (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, (1, 1), stride=(1, 1)), 
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 

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
    def __init__(self, emb_size: int, num_heads: int, drop_p: float = 0.5, forward_expansion: int = 4, forward_drop_p: float = 0.3):
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


class BiMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv
        )

        self.mamba_rev = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln1_rev = nn.LayerNorm(d_model)
        self.ln2_rev = nn.LayerNorm(d_model)


        self.ffn = FeedForwardBlock(d_model, 4, dropout)
        self.ffn_rev = FeedForwardBlock(d_model, 4, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward_branch(self, x, mamba, ln1, ln2, ffn, flip_time=False):
        if flip_time:
            x_in = torch.flip(x, dims=[1])  # (B, S, D)
        else:
            x_in = x

        y = mamba(x_in)                           # Mamba
        y = self.dropout(y)

        if flip_time:
            y = torch.flip(y, dims=[1])
        y = ln1(x + y)                            # Add & Norm

        y2 = ffn(y)                                # FeedForward
        y2 = self.dropout(y2)
        y = ln2(y + y2)                            # Add & Norm

        return y

    def forward(self, x):
        out_fwd = self.forward_branch(
            x, self.mamba_fwd, self.ln1, self.ln2, self.ffn, flip_time=False
        )

        out_rev = self.forward_branch(
            x, self.mamba_rev, self.ln1_rev, self.ln2_rev, self.ffn_rev, flip_time=True
        )

        out = 0.5 * (out_fwd + out_rev) 
        return out


class BiMambaEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, **block_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [BiMambaBlock(d_model=d_model, **block_kwargs) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_ln(x)


class CrossAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, drop_p: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(drop_p)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:

        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        # Keys and values from source/context sequence
        keys = rearrange(self.keys(context), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(context), "b n (h d) -> b h n d", h=self.num_heads)

        # Attention: target attends to source
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
                
        scaling = self.emb_size ** (1 / 2)
        att = torch.nn.functional.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        
        # Weighted sum of values
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class FusionLayer(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, drop_p: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.att_norm = nn.LayerNorm(emb_size)
        self.att = CrossAttention(emb_size, num_heads, drop_p=0.5)
        self.att_dropout = nn.Dropout(drop_p)
        self.ffn_norm = nn.LayerNorm(emb_size)
        self.ffn = FeedForwardBlock(emb_size, expansion=4, drop_p=0.3)
        self.ffn_dropout = nn.Dropout(drop_p)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.att_norm(x)
        x = self.att(x, context)
        x = self.att_dropout(x)
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        return x



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


class HSST(nn.Module):
    def __init__(self, emb_size: int = 40, depth: int = 2, num_heads: int = 4, n_channels: int = 18, n_classes: int = 1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, n_channels)
        self.transformer = TransformerEncoder(depth, emb_size, num_heads)
        self.mamba = BiMambaEncoder(emb_size, depth)
        self.fusion = FusionLayer(emb_size, num_heads, drop_p=0.3)
        self.clshead = ClassificationHead(emb_size, n_classes)
    
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        emb = self.patch_embedding(x)
        transformer_out = self.transformer(emb)
        mamba_out = self.mamba(emb)
        fusion_out = self.fusion(transformer_out, mamba_out)
        fusion_out += mamba_out
        out = self.clshead(fusion_out)
        return out
    

if __name__ == '__main__':
    import time
    model = HSST(emb_size=40, depth=4, num_heads=4, n_channels=18, n_classes=1).to('cuda')
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
    sample = torch.randn(1, 18, 2560).to('cuda')
    t0 = time.time()
    out = model(sample)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(f"Output shape: {out.shape}")
    