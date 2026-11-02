"""Spatio-Temporal U-Net alternative architecture with identical sequence-to-sequence interface."""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double 2D Convolution with Group Normalization and LeakyReLU."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_c),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpatioTemporalUNet(nn.Module):
    """Spatiotemporal U-Net for multi-horizon wildfire progression forecasting.
    
    Accepts (Batch, Seq_Len=5, In_Channels=6, H=128, W=128) and outputs (Batch, Pred_Len=3, 1, H, W).
    """

    def __init__(
        self,
        in_channels: int = 6,
        seq_len: int = 5,
        pred_len: int = 3,
        out_channels: int = 1,
        base_features: int = 32,
        use_intensity_head: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.out_channels = out_channels
        self.use_intensity_head = use_intensity_head

        # Total input channels collapsed across time
        in_total = in_channels * seq_len  # 6 * 5 = 30 channels

        # Encoder
        self.enc1 = ConvBlock(in_total, base_features)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_features * 2, base_features * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_features * 2, base_features)

        # Heads outputting (Pred_Len * Out_Channels)
        out_dim = pred_len * out_channels
        self.presence_out = nn.Sequential(
            nn.Conv2d(base_features, out_dim, kernel_size=1),
            nn.Sigmoid()
        )

        if self.use_intensity_head:
            self.intensity_out = nn.Sequential(
                nn.Conv2d(base_features, out_dim, kernel_size=1),
                nn.ReLU()
            )

    def forward(
        self,
        x: torch.Tensor,
        pred_steps: Optional[int] = None,
        return_intensity: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = x.shape
        # Flatten time and channel dimensions: (B, T*C, H, W)
        x_flat = x.view(b, t * c, h, w)

        # Encoder pass
        e1 = self.enc1(x_flat)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b_feat = self.bottleneck(p2)

        # Decoder pass with skip connections
        d2 = self.up2(b_feat)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Heads
        presence_flat = self.presence_out(d1) # (B, pred_len * out_c, H, W)
        presence = presence_flat.view(b, self.pred_len, self.out_channels, h, w)

        if return_intensity and self.use_intensity_head:
            intensity_flat = self.intensity_out(d1)
            intensity = intensity_flat.view(b, self.pred_len, self.out_channels, h, w)
            return presence, intensity

        return presence
