import torch 
import torch.nn as nn
import math
from einops import rearrange


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
    def __init__(self, emb_size: int, n_channels: int, n_fft: int, hop_length: int = 100):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

        self.patch_embedding = nn.Linear(n_fft // 2 + 1, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)


    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i: i + 1, :])
            channel_spec_emb = self.patch_embedding(rearrange(channel_spec_emb, 'b f t -> b t f'))
            B, T, _ = channel_spec_emb.shape

            channel_token_emb = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(B, T, 1)
            )

            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)
            emb_seq.append(channel_emb)
        
        emb = torch.cat(emb_seq, dim=1)
        return emb


if __name__ == '__main__':
    model = PatchFrequencyEmbedding(emb_size=64, n_channels=16, n_fft = 200, hop_length = 100)
    input = torch.randn(5, 16, 2560)
    out = model(input)
    print(out.shape)