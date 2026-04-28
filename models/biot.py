import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer

class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size, n_freq):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, drop_p: float = 0.1, max_len: int = 1024):
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

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class BIOTEncoder(nn.Module):
    def __init__(self, emb_size=64, num_heads=4, depth=4, num_channels=16, n_fft=200, hop_length=100):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(self.n_fft))

        self.patch_embedding = PatchFrequencyEmbedding(emb_size=emb_size, n_freq=self.n_fft // 2 + 1)
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=num_heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        self.channel_tokens = nn.Embedding(num_channels, emb_size)
        self.index = nn.Parameter(torch.LongTensor(range(num_channels)), requires_grad=False)
        
    def stft(self, sample):
        sample = sample.float()
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            window=self.window,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        emb = torch.cat(emb_seq, dim=1)
        emb = self.transformer(emb).mean(dim=1)
        return emb

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, num_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out
    
# supervised classifier module
class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=64, depth=4, num_heads=4, num_classes=1, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, num_heads=num_heads, depth=depth, **kwargs)
        self.classifier = ClassificationHead(emb_size, num_classes)

    def forward(self, x):
        x = self.biot(x)
        x = self.classifier(x)
        return x

