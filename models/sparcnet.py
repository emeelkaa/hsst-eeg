import torch
import torch.nn as nn
import math


class DenseLayer(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            growth_rate: int, 
            bottleneck_size: int, 
            dropout: float = 0.5,
            conv_bias: bool = True, 
            batch_norm: bool = True,
            activation = nn.ELU,
            kernel_size_conv1: int = 1, 
            kernel_size_conv2: int = 3,
            stride_conv1: int = 1,  
            stride_conv2: int = 1, 
            padding_conv2: int = 1
    ):
        super().__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(in_channels)
            self.batch_norm2 = nn.BatchNorm1d(bottleneck_size * growth_rate)

        self.elu1 = activation()
        self.conv1d = nn.Conv1d(in_channels, bottleneck_size * growth_rate, kernel_size=kernel_size_conv1, stride=stride_conv1, bias=conv_bias)
        self.elu2 = activation()
        self.conv2d = nn.Conv1d(bottleneck_size * growth_rate, growth_rate, kernel_size=kernel_size_conv2, stride=stride_conv2, padding=padding_conv2, bias=conv_bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = x 
        if self.batch_norm:
            out = self.batch_norm1(out)
        out = self.elu1(out)
        out = self.conv1d(out)
        if self.batch_norm:
            out = self.batch_norm2(out)
        out = self.elu2(out)
        out = self.conv2d(out)
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Sequential):
    def __init__(
            self, 
            num_layers: int, 
            in_channels: int, 
            bottleneck_size: int, 
            growth_rate: int, 
            dropout: float = 0.5, 
            conv_bias: bool = True, 
            batch_norm: bool = True, 
            activation = nn.ELU, 
            kernel_size_conv1: int = 1, 
            kernel_size_conv2: int = 3, 
            stride_conv1: int = 1, 
            stride_conv2: int = 1, 
            padding_conv2: int = 1
    ):
        super().__init__()
        for idx_layer in range(num_layers):
            layer = DenseLayer(
                in_channels=in_channels + idx_layer * growth_rate,
                growth_rate=growth_rate,
                bottleneck_size=bottleneck_size,
                dropout=dropout,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
                activation=activation,
                kernel_size_conv1=kernel_size_conv1,
                kernel_size_conv2=kernel_size_conv2,
                stride_conv1=stride_conv1,
                stride_conv2=stride_conv2,
                padding_conv2=padding_conv2
            )
            self.add_module(f"denselayer{idx_layer + 1}", layer)


class TransitionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_bias: bool = True,
        batch_norm: bool = True,
        activation = nn.ELU,
        kernel_size_trans: int = 2,
        stride_trans:int = 2,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm = nn.BatchNorm1d(in_channels)
        self.activation = activation()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=conv_bias)
        self.pool = nn.AvgPool1d(kernel_size=kernel_size_trans, stride=stride_trans)
    
    def forward(self, x):
        if self.batch_norm:
            x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class SPARCNet(nn.Module):
    def __init__(
            self, 
            num_channels: int, 
            num_times: int, 
            num_classes: int,
            block_layers: int = 4,
            growth_rate: int = 16,
            bottleneck_size: int = 16, 
            dropout: float = 0.5,
            conv_bias: bool = True,
            batch_norm: bool = True,
            activation = nn.ELU,
            kernel_size_conv0: int = 7,
            kernel_size_conv1: int = 1,
            kernel_size_conv2: int = 3,
            kernel_size_pool: int = 3,
            stride_pool: int = 2,
            stride_conv0: int = 2,
            stride_conv1: int = 1,
            stride_conv2: int = 1,
            padding_pool: int = 1,
            padding_conv0: int = 3,
            padding_conv2: int = 1,
            kernel_size_trans: int = 2,
            stride_trans: int = 2,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_times = num_times
        self.num_classes = num_classes
        out_channels = 2 ** (math.floor(math.log2(num_channels)) + 1)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size_conv0, 
                      stride=stride_conv0, 
                      padding=padding_conv0, 
                      bias=conv_bias),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=kernel_size_pool, stride=stride_pool, padding=padding_pool)
        )

        num_channels = out_channels
        for n_layer in range(math.floor(math.log2(self.num_times // 4))):
            block = DenseBlock(
                num_layers=block_layers,
                in_channels=num_channels, 
                growth_rate=growth_rate, 
                bottleneck_size=bottleneck_size, 
                dropout=dropout, 
                conv_bias=conv_bias, 
                batch_norm=batch_norm, 
                activation=activation, 
                kernel_size_conv1=kernel_size_conv1, 
                kernel_size_conv2=kernel_size_conv2, 
                stride_conv1=stride_conv1, 
                stride_conv2=stride_conv2, 
                padding_conv2=padding_conv2
            )
            self.encoder.add_module(f"denseblock{n_layer + 1}", block)
            num_channels = num_channels + block_layers * growth_rate

            trans = TransitionLayer(
                in_channels=num_channels,
                out_channels=num_channels // 2,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
                activation=activation,
                kernel_size_trans=kernel_size_trans,
                stride_trans=stride_trans,
            )
            self.encoder.add_module(f"transition{n_layer + 1}", trans)
            num_channels = num_channels // 2
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.activation_layer = activation()
        self.flatten_layer = nn.Flatten()
        self.final_layer = nn.Linear(num_channels, self.num_classes)

    
    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        emb = self.adaptive_pool(emb)
        emb = self.activation_layer(emb)
        emb = self.flatten_layer(emb)
        out = self.final_layer(emb)
        return out


if __name__ == "__main__":
    import time 
    model = SPARCNet(16, 2560, 1).to('cuda')
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    sample = torch.randn(1, 16, 2560).to('cuda')
    t0 = time.time()
    out = model(sample)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(f"Output shape: {out.shape}")