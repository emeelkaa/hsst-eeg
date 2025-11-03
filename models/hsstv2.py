import torch
import torch.nn as nn
import math
from einops import rearrange
from einops.layers.torch import Reduce
from mamba_ssm import Mamba
from models import TransformerEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, drop_p: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_p)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    

class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size: int = 64, n_channels: int = 16, sfreq: int = 250):
        super().__init__()
        self.n_fft = sfreq
        self.hop_length = sfreq // 2
        self.patch_embedding = nn.Linear(self.n_fft // 2 + 1, emb_size)

        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

        self.positional_encoding = PositionalEncoding(emb_size)

        self.register_buffer("window", torch.hann_window(self.n_fft))
        
    def stft(self, sample):
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)
    
    def forward(self, x: torch.Tensor):
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(
                rearrange(channel_spec_emb, "b f t -> b t f")
            )
            B, T, D = channel_spec_emb.shape

            channel_token_emb = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(B, T, 1)
            )

            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb )
            emb_seq.append(channel_emb) 
        emb = torch.stack(emb_seq, dim=1)
        return emb


class BiMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, drop_p: float = 0.3):
        super().__init__()

        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.mamba_rev = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln1_rev = nn.LayerNorm(d_model)
        self.ln2_rev = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(drop_p),
        )

        self.ffn_rev = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(drop_p),
        )

        self.dropout = nn.Dropout(drop_p)

    def forward_branch(self, x, mamba, ln1, ln2, ffn, flip_time=False):
        if flip_time:
            x_in = torch.flip(x, dims=[1]) 
        else:
            x_in = x

        y = mamba(x_in)                           
        y = self.dropout(y)

        if flip_time:
            y = torch.flip(y, dims=[1])
        y = ln1(x + y)                            

        y2 = ffn(y)                                
        y2 = self.dropout(y2)
        y = ln2(y + y2)                      
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
    def __init__(self, emb_size: int = 64, depth: int = 2, num_heads: int = 4, n_channels: int = 16, sfreq: int = 256, n_classes: int = 1):
        super().__init__()
        self.patch_embedding = PatchFrequencyEmbedding(emb_size, n_channels, sfreq)
        self.transformer = TransformerEncoder(depth, emb_size, num_heads)
        self.mamba = BiMambaEncoder(emb_size, depth)
        #self.caf = CAF(emb_size, num_heads)
        self.cls_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        f_e = self.patch_embedding(x)

        f_t = rearrange(f_e, "b c t d -> (b t) c d")
        f_t = self.transformer(f_t)
        f_t = rearrange(f_t, "(b t) c d -> b (c t) d", b=x.shape[0])

        f_m = rearrange(f_e, "b c t d -> (b c) t d")
        f_m = self.mamba(f_m)
        f_m = rearrange(f_e, "b c t d -> b (c t) d", b=x.shape[0])

        out = 0.5 * (f_t + f_m)
        out = self.cls_head(out)
        return out


if __name__ == "__main__":
    import time
    model = HSST().to('cuda')
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    x = torch.randn(5, 16, 2560).to('cuda')
    t0 = time.time()
    emb = model(x)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")

    print(emb.shape)