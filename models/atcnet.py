import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Reduce

class ConvBlock(nn.Module):
    def __init__(
            self,
            num_channels: int,
            num_filters: int = 16,
            kernel_size_1: int = 64,
            kernel_size_2: int = 16,
            pool_size_1: int = 8,
            pool_size_2: int = 7,
            depth_mult: int = 2,
            dropout: float = 0.3
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=num_filters, 
            kernel_size=(kernel_size_1, 1), 
            padding="same",
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
    
        num_depth_kernels = num_filters * depth_mult

        self.conv2 = nn.Conv2d(
            in_channels=num_filters, 
            out_channels=num_depth_kernels, 
            groups=num_filters,
            kernel_size=(1, num_channels),
            padding="valid",
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_depth_kernels)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(pool_size_1, 1))
        self.drop2 = nn.Dropout2d(dropout)

        self.conv3 = nn.Conv2d(
            in_channels=num_depth_kernels,
            out_channels=num_depth_kernels,
            kernel_size=(kernel_size_2, 1),
            padding="same",
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_depth_kernels)
        self.activation3 = nn.ELU()
        self.pool3 = nn.AvgPool2d(kernel_size=(pool_size_2, 1))
        self.drop3 = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation3(x)
        x = self.pool3(x)
        out = self.drop3(x)
        return out


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


class AttentionBlock(nn.Sequential):
    def __init__(self, emb_size: int, num_heads: int, drop_p: float = 0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )))


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            padding=(kernel_size - 1) * dilation, 
            **kwargs
        )
    
    def forward(self, x):
        out = F.conv1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return out[..., : -self.padding[0]]


class TCNResidualBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            kernel_size: int = 4,
            n_filters: int = 32,
            dropout: float = 0.3,
            activation = nn.ELU(),
            dilation: int = 1
    ):
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels=in_channels, 
            out_channels=n_filters, 
            kernel_size=kernel_size, 
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.activation1 = activation
        self.conv2 = CausalConv1d(
            in_channels=n_filters, 
            out_channels=n_filters, 
            kernel_size=kernel_size, 
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.activation2 = activation

        if in_channels != n_filters:
            self.reshaping_conv = nn.Conv1d(
                in_channels=in_channels,  # Specify input channels
                out_channels=n_filters,  # Specify output channels
                kernel_size=1,
                padding="same",
            )
        else:
            self.reshaping_conv = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation2(out)

        out = out + self.reshaping_conv(x)
        
        return out 


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes: int = 1):
        super().__init__()
        self.emb_size = emb_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Dropout(0.3),  
            nn.Linear(emb_size, n_classes)
        )   
  
    def forward(self, x):
        out = self.classifier(x)
        return out

class ATCNet(nn.Module):
    def __init__(
            self,
            num_channels: int,
            num_times: int,
            num_classes: int,
            conv_block_num_filters: int = 16,
            conv_block_kernel_size_1: int = 64,
            conv_block_kernel_size_2: int = 16,
            conv_block_pool_size_1: int = 8,
            conv_block_pool_size_2: int = 7,
            conv_block_depth_mult: int = 2,
            conv_block_dropout: float = 0.3,
            num_heads: int = 4,
            num_windows: int = 4,
            tcn_n_filters: int = 32,
            tcn_kernel_size: int = 4,
            tcn_dropout: float = 0.3,
            tcn_depth: int = 2
    ):
        super().__init__()

        self.F2 = int(conv_block_depth_mult * conv_block_num_filters)
        self.Tc = int(num_times / (conv_block_pool_size_1 * conv_block_pool_size_2))
        self.Tw = self.Tc - num_windows + 1

        self.conv_block = ConvBlock(
            num_channels=num_channels,
            num_filters=conv_block_num_filters,
            kernel_size_1=conv_block_kernel_size_1,
            kernel_size_2=conv_block_kernel_size_2,
            pool_size_1=conv_block_pool_size_1,
            pool_size_2=conv_block_pool_size_2,
            depth_mult=conv_block_depth_mult,
            dropout=conv_block_dropout
        )

        self.attention_block = nn.ModuleList([
            AttentionBlock(emb_size=self.F2, num_heads=num_heads)
            for _ in range(num_windows)
        ])

        self.temporal_conv_nets = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        TCNResidualBlock(
                            in_channels=self.F2 if i == 0 else tcn_n_filters,
                            kernel_size=tcn_kernel_size,
                            n_filters=tcn_n_filters,
                            dropout=tcn_dropout,
                            activation=nn.ELU(),
                            dilation=2**i,
                        )
                        for i in range(tcn_depth)
                    ]
                )
                for _ in range(num_windows)
            ]
        )

        self.cls = ClassificationHead(emb_size=tcn_n_filters * num_windows, n_classes=num_classes)
        

        

    def forward(self, x):
        x = rearrange(x, "b c t -> b t c")
        x = x.unsqueeze(1)
        B, _, C, T = x.shape
        x = self.conv_block(x)
        x = rearrange(x, 'b f2 tc 1 -> b f2 tc')
        
        sw_concat = []

        for idx, (attention, tcn_module) in enumerate(zip(self.attention_block, self.temporal_conv_nets)):
            x_i = x[..., idx : idx + self.Tw]
            x_i = rearrange(x_i, 'b f2 t -> b t f2')  # Transpose before attention
            att = attention(x_i)
            att = rearrange(att, 'b t f2 -> b f2 t')
            tcn = tcn_module(att)[..., -1]
            sw_concat.append(tcn)
        
        sw_concat_agg = torch.cat(sw_concat, dim=1)

        out = self.cls(sw_concat_agg)
    
        return out


if __name__ == "__main__":
    import torch
    import time
    model = ATCNet(num_channels=16, num_times=2560, num_classes=1).to('cuda')
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    sample = torch.randn(1, 16, 2560).to('cuda')
    t0 = time.time()
    out = model(sample)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(f"Output shape: {out.shape}")