
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from linear_attention_transformer import LinearAttentionTransformer
from einops import rearrange
from einops.layers.torch import Reduce

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, drop_p=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_p)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size, num_channels, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft 
        self.hop_length = hop_length
        self.segment_emb = nn.Linear((n_fft // 2 + 1), emb_size)

        self.position_emb = PositionalEncoding(emb_size)
        self.channel_emb = nn.Embedding(num_channels, emb_size)
        self.index = nn.Parameter(torch.LongTensor(range(num_channels)), requires_grad=False)

        self.register_buffer('window', torch.hann_window(n_fft))
    
    def stft(self, x):
        x_flat = rearrange(x, 'b c t -> (b c) t')
        spectral = torch.stft( 
            input = x_flat,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            window=self.window,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x):
        B, C, T = x.shape

        emb = self.stft(x)
        emb = self.segment_emb(rearrange(emb, 'B D T -> B T D'))
        emb = rearrange(emb, '(B C) T D -> B C T D', B=B, C=C)

        ch_emb = self.channel_emb(self.index)
        ch_emb = ch_emb[None, :, None, :].expand(B, -1, emb.shape[2], -1)
        emb = emb + ch_emb
        emb = self.position_emb(rearrange(emb, 'b c t d -> (b c) t d'))

        return emb

class MambaBlock(nn.Module):
    def __init__(self, emb_size, d_state=32, d_conv=4, dropout=0.2):
        super().__init__()

        self.mamba_fwd = Mamba(d_model=emb_size, d_state=d_state, d_conv=d_conv)
        self.mamba_rev = Mamba(d_model=emb_size, d_state=d_state, d_conv=d_conv)

        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        self.ln1_rev = nn.LayerNorm(emb_size)
        self.ln2_rev = nn.LayerNorm(emb_size)

        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 4, emb_size),
        )

        self.ffn_rev = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 4, emb_size),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward_branch(self, x, mamba, ln1, ln2, ffn, flip_time=False):
        if flip_time:
            x_in = torch.flip(x, dims=[1])
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
    
class HSST(nn.Module):
    def __init__(self, emb_size, depth, num_heads, num_channels, num_classes, n_fft=200, hop_length=100):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, num_channels, n_fft, hop_length)

        self.attention = LinearAttentionTransformer(
            dim=emb_size,
            heads=num_heads,
            depth=2,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  
            attn_dropout=0.2,
        )

        self.mamba = nn.ModuleList([
            MambaBlock(emb_size) for _ in range(2)
        ])
        self.cls_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )
    
    def forward(self, x):
        f_e = self.patch_embedding(x)
        for mamba in self.mamba:
            f_e = mamba(f_e)
        f_e = rearrange(f_e, '(b c) t d -> b (c t) d', b=x.shape[0])
        f_e = self.attention(f_e)
        out = self.cls_head(f_e)
        return out
