"""
model.py
--------
3-tier U-Net architecture for binary segmentation of breast-cancer cell
nuclei in H&E-stained histopathology images (Capstone Project 2).

Input:   RGB image, 256x256
Output:  single-channel probability mask, 256x256 (sigmoid applied
         at loss time via BCEWithLogitsLoss)

The architecture keeps the classic U-Net shape but is shallow enough
to train on only 46 samples without overfitting:

    Encoder    64  → 128 → 256
    Bottleneck              → 512
    Decoder    256 ← 128 ← 64  (with skip connections from encoder)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convs each followed by BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """DoubleConv → MaxPool. Returns (features_before_pool, features_after_pool)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        feat = self.conv(x)
        return feat, self.pool(feat)


class Up(nn.Module):
    """Upsample → concat with skip → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """3-tier U-Net: 64 → 128 → 256 → 512 (bottleneck)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.d1 = Down(in_channels, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.u3 = Up(512, 256, 256)
        self.u2 = Up(256, 128, 128)
        self.u1 = Up(128, 64, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, x = self.d1(x)   # 64
        s2, x = self.d2(x)   # 128
        s3, x = self.d3(x)   # 256
        x = self.bottleneck(x)   # 512
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        return self.out_conv(x)   # raw logits


if __name__ == "__main__":
    net = UNet()
    dummy = torch.randn(2, 3, 256, 256)
    out = net(dummy)
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Output shape: {tuple(out.shape)}  (expect [2, 1, 256, 256])")
