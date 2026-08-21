import torch
import torch.nn as nn
import os
from pathlib import Path

# ------------ Blocks --------------
class RC_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.sigma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def fast_erf(self, x):
        return torch.tanh(1.702 * x)

    def forward(self, x):
        x_centered = (x - self.mu) / (self.sigma + 1e-5)
        phi = 0.5 * (1 + self.fast_erf(x_centered / 1.4142))
        weight = torch.where(x > self.mu, 1 - phi, phi)
        return x + x * weight

class SERegNetBottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride=1, se_ratio=0.25, group_width=8, use_rc=True, is_last_block=False):
        super().__init__()
        self.use_rc = use_rc and is_last_block and out_c >= 256  

        mid_c = out_c
        groups = mid_c // group_width

        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_c, mid_c, kernel_size=3, stride=stride, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)

        se_c = max(1, int(out_c * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, se_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_c, out_c, kernel_size=1),
            nn.Hardsigmoid(inplace=True)
        )

        if self.use_rc:
            self.rc = RC_Layer(out_c)

        if in_c != out_c or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.skip = nn.Identity()

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out * self.se(out)
        if self.use_rc:
            out = self.rc(out)
        return self.relu_out(out + self.skip(x))

# ------------ Unified Single-Head Model --------------
class BestenSingle(nn.Module): 
    def __init__(
        self,
        num_classes, 
        # Unified layer configuration: (repeats, output_channels, stride)
        layer_config = [(1, 64, 2), (1, 128, 2), (1, 256, 2), (1, 512, 2)],
        se_ratio=0.25, 
        group_width=8,                 
        use_rc=True
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        # Generate the unified sequential stages
        self.stages, final_out_c = self._make_stage(
            24, layer_config, se_ratio, group_width, use_rc)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_out_c, num_classes)

    def _make_stage(self, in_c, config, se_ratio, group_width, use_rc):
        layers = []
        for repeat, out_c, stride in config:
            for i in range(repeat):
                block_stride = stride if i == 0 else 1
                is_last = (i == repeat - 1)
                layers.append(SERegNetBottleneck(
                    in_c if i == 0 else out_c,
                    out_c,
                    stride=block_stride,
                    se_ratio=se_ratio,
                    group_width=group_width,
                    use_rc=use_rc,
                    is_last_block=is_last
                ))
            in_c = out_c
        return nn.Sequential(*layers), in_c

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        out = self.fc(self.pool(x).flatten(1))
        return out
