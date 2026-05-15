"""
FPN v2 with Ghost module support (from NanoDet-Plus).

Two modes:
- Standard: depthwise-separable convs (for tiny config)
- Ghost-PAN: Ghost modules for efficient feature fusion (for small config)

Ghost module: generates feature maps via cheap linear operations from a
small set of "intrinsic" features. Gets ~2x param reduction with <1% mAP loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class GhostModule(nn.Module):
    """
    Ghost module from GhostNet / NanoDet Ghost-PAN.

    Generates feature maps in two steps:
    1. Primary: standard conv to generate a small set of intrinsic features
    2. Cheap: depthwise conv on intrinsic features to generate "ghost" features
    3. Concat primary + ghost features

    Result: same output channels as a standard conv, but ~2x fewer params/FLOPs.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 ratio: int = 2, dw_kernel: int = 3, stride: int = 1):
        super().__init__()
        self.out_channels = out_channels
        intrinsic_channels = math.ceil(out_channels / ratio)
        ghost_channels = intrinsic_channels * (ratio - 1)

        # Primary convolution (generates intrinsic features)
        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, intrinsic_channels, kernel_size,
                      stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(intrinsic_channels),
            nn.SiLU(inplace=True),
        )

        # Cheap operation (generates ghost features from intrinsic)
        self.cheap = nn.Sequential(
            nn.Conv2d(intrinsic_channels, ghost_channels, dw_kernel,
                      stride=1, padding=dw_kernel // 2,
                      groups=intrinsic_channels, bias=False),
            nn.BatchNorm2d(ghost_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        ghost = self.cheap(primary)
        out = torch.cat([primary, ghost], dim=1)
        return out[:, :self.out_channels, :, :]  # Trim to exact channel count


class GhostBottleneck(nn.Module):
    """Ghost bottleneck: Ghost module → DW conv (optional) → Ghost module."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int,
                 stride: int = 1):
        super().__init__()
        self.ghost1 = GhostModule(in_channels, mid_channels)

        # Depthwise conv for spatial mixing
        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride=stride,
                      padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        ) if stride > 1 else nn.Identity()

        self.ghost2 = GhostModule(mid_channels, out_channels)

        # Shortcut
        self.shortcut = nn.Identity() if (stride == 1 and in_channels == out_channels) else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        out = self.ghost1(x)
        out = self.dw_conv(out)
        out = self.ghost2(out)
        return out + shortcut


class LightFPN(nn.Module):
    """
    Lightweight top-down Feature Pyramid Network.

    Supports two modes:
    - Standard: depthwise-separable smoothing convs
    - Ghost: Ghost module-based smoothing (from NanoDet Ghost-PAN)

    Top-down pathway only (no bottom-up PAN path).
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 48,
                 use_ghost: bool = True):
        """
        Args:
            in_channels_list: Channel counts from backbone, len 3 or 4
                              (3 for [C3,C4,C5], 4 for [C2,C3,C4,C5]).
            out_channels: Unified output channels for all FPN levels
            use_ghost: Use Ghost modules instead of DW-sep convs
        """
        super().__init__()
        assert len(in_channels_list) in (3, 4), (
            f"FPN expects 3 or 4 input levels, got {len(in_channels_list)}"
        )
        self.num_levels = len(in_channels_list)

        # Lateral 1x1 convolutions per level (ordered low-stride → high-stride)
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])

        # Smoothing after fusion
        if use_ghost:
            self.smooths = nn.ModuleList([
                GhostBottleneck(out_channels, out_channels, out_channels)
                for _ in range(self.num_levels)
            ])
        else:
            self.smooths = nn.ModuleList([
                DepthwiseSeparableConv(out_channels, out_channels)
                for _ in range(self.num_levels)
            ])

        self.out_channels = out_channels
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [C3, C4, C5] or [C2, C3, C4, C5] from backbone
                      (ordered low-stride → high-stride)

        Returns:
            Per-level outputs in the same order, all with `out_channels` channels.
        """
        assert len(features) == self.num_levels

        # Lateral projections
        pyramid = [lat(f) for lat, f in zip(self.laterals, features)]

        # Top-down fusion: highest-stride → lowest. Add upsampled coarser level.
        for i in range(self.num_levels - 1, 0, -1):
            pyramid[i - 1] = pyramid[i - 1] + F.interpolate(
                pyramid[i], size=pyramid[i - 1].shape[2:], mode="nearest"
            )

        # Smooth each level
        pyramid = [smooth(p) for smooth, p in zip(self.smooths, pyramid)]
        return pyramid
