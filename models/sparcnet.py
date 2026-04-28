import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, input_channels, expansion, bn_size, drop_p, conv_bias, batch_norm):
        super().__init__()
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm1 = nn.BatchNorm1d(input_channels)
            self.norm2 = nn.BatchNorm1d(bn_size * expansion)
        self.elu1 = nn.ELU()
        self.conv1 = nn.Conv1d(
            input_channels, 
            bn_size * expansion, 
            kernel_size=1, 
            stride=1, 
            bias=conv_bias
        )
        self.elu2 = nn.ELU()
        self.conv2 = nn.Conv1d(
            bn_size * expansion, 
            expansion, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=conv_bias
        )
        self.drop_p = drop_p
    
    def forward(self, x):
        if self.batch_norm:
            new_features = self.norm1(x)
        else:
            new_features = x 
        new_features = self.elu1(new_features)
        new_features = self.conv1(new_features)
        if self.batch_norm:
            new_features = self.norm2(new_features)
        new_features = self.elu2(new_features)
        new_features = self.conv2(new_features)
        new_features = F.dropout(new_features, p=self.drop_p, training=self.training)
        return torch.cat([x, new_features], 1)
    
class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_channels, expansion, bn_size, drop_p, conv_bias, batch_norm):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for idx_layer in range(num_layers):
            layer = DenseLayer(
                input_channels + idx_layer * expansion,
                expansion, 
                bn_size,
                drop_p, 
                conv_bias, 
                batch_norm,
            )
            self.layers.append(layer)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, conv_bias, batch_norm):
        super().__init__()
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm = nn.BatchNorm1d(input_channels)
        self.elu = nn.ELU()
        self.conv = nn.Conv1d(
            input_channels, 
            output_channels, 
            kernel_size=1, 
            stride=1, 
            bias=conv_bias
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        if self.batch_norm:
            x = self.norm(x)
        x = self.elu(x)
        x = self.conv(x)
        x = self.pool(x)    
        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.elu = nn.ELU()
        self.fc = nn.Linear(input_channels, num_classes)
    
    def forward(self, x):
        x = x.squeeze(-1)
        x = self.elu(x)
        x = self.fc(x)
        return x

class SPaRCNet(nn.Module):
    def __init__(
        self,     
        num_channels=16, 
        num_timepoints=2560,
        num_classes=1, 
        block_layers=4, 
        expansion=16, 
        bn_size=16, 
        drop_p=0.5, 
        conv_bias=True, 
        batch_norm=True
    ):
        super().__init__()
        out_channels = 2 ** (math.floor(np.log2(num_channels)) + 1)
        
        self.conv0 = nn.Conv1d(
            num_channels, 
            out_channels, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=conv_bias
        )
        self.norm0 = nn.BatchNorm1d(out_channels)
        self.elu0 = nn.ELU()
        self.pool0 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        input_channels = out_channels

        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for n_layer in np.arange(math.floor(np.log2(num_timepoints // 4))):
            block = DenseBlock(
                num_layers=block_layers,
                input_channels=input_channels,
                expansion=expansion,
                bn_size=bn_size,
                drop_p=drop_p,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.dense_blocks.append(block)
            input_channels = input_channels + block_layers * expansion

            trans = TransitionLayer(
                input_channels=input_channels,
                output_channels=input_channels // 2,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.transitions.append(trans)
            input_channels = input_channels // 2
        
        self.classifier = ClassificationHead(input_channels, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        # initial conv block
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.elu0(x)
        x = self.pool0(x)
        
        # dense blocks and transitions
        for block, trans in zip(self.dense_blocks, self.transitions):
            x = block(x)
            x = trans(x)
        
        # classification
        out = self.classifier(x)
        return out

if __name__ == "__main__":
    import time
    num_timepoints = 720000
    model = SPaRCNet(num_timepoints=num_timepoints).to('cuda')
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    x = torch.randn(1, 16, num_timepoints).to('cuda')
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(f"Output shape: {out.shape}")