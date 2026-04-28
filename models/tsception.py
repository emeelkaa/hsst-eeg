import torch 
import torch.nn as nn

class TSception(nn.Module):
    def conv_block(self, in_channels, out_channels, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_channels, sfreq, num_classes, num_T=32, num_S=64, hidden=64, dropout_rate=0.3):
        super().__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel=(1,length) and the stride=1 we can use conv2d as conv1d
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sfreq)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sfreq)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sfreq)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (num_channels, 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(num_channels * 0.5), 1), (int(num_channels * 0.5), 1), int(self.pool*0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    import time
    model = TSception(num_channels=16, sfreq=200, num_classes=1).to('cuda')
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    x = torch.randn(1, 16, 720000).to('cuda')
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(f"Output shape: {out.shape}")
    