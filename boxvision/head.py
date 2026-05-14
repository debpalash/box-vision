"""
Detection head v2: minimal, no centerness, no DFL.

Outputs per FPN level:
  - objectness: 1 channel (is there an object here?)
  - bbox_reg:   4 channels (l, t, r, b distances)
  - class_logits: num_classes channels, only when num_classes > 1
  - centerness: 1 channel, only when use_centerness=True

When num_classes == 1 the model behaves like the original class-agnostic
BoxVision. When num_classes > 1 a class prediction head is added; final
detection score = sigmoid(objectness) * sigmoid(class_logits[k]).
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional

from .fpn import DepthwiseSeparableConv


class ScaleLayer(nn.Module):
    """Learnable scale parameter for bbox regression per FPN level."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FCOSHead(nn.Module):
    """
    Minimal FCOS-style detection head — class-agnostic, no centerness.

    For each spatial location on each FPN level, predicts:
        - objectness: 1 value (is there an object here?)
        - bbox: 4 values (l, t, r, b distances to box edges)
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_convs: int = 1,
        use_depthwise: bool = True,
        use_centerness: bool = False,
        num_levels: int = 3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.use_centerness = use_centerness
        self.num_classes = num_classes
        self.multi_class = num_classes > 1

        ConvBlock = DepthwiseSeparableConv if use_depthwise else _StandardConvBlock

        # Objectness branch (shared by class head when multi-class — keeps params small)
        obj_layers = []
        for _ in range(num_convs):
            obj_layers.append(ConvBlock(in_channels, in_channels))
        self.obj_tower = nn.Sequential(*obj_layers)
        self.obj_pred = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # Class branch (only when multi-class). Reuses obj_tower features.
        if self.multi_class:
            self.class_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)

        # BBox regression branch
        bbox_layers = []
        for _ in range(num_convs):
            bbox_layers.append(ConvBlock(in_channels, in_channels))
        self.bbox_tower = nn.Sequential(*bbox_layers)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)

        # Optional centerness (disabled by default in v2)
        if self.use_centerness:
            self.centerness_pred = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # Per-level learnable scale for bbox regression
        self.scales = nn.ModuleList([ScaleLayer(1.0) for _ in range(num_levels)])

        self._init_weights()

    def _init_weights(self):
        for module in [self.obj_tower, self.bbox_tower]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Objectness bias init for focal loss stability
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.normal_(self.obj_pred.weight, std=0.01)
        nn.init.constant_(self.obj_pred.bias, bias_value)

        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.zeros_(self.bbox_pred.bias)

        if self.multi_class:
            # Match objectness prior for class logits — start each class near 0.01
            nn.init.normal_(self.class_pred.weight, std=0.01)
            nn.init.constant_(self.class_pred.bias, bias_value)

        if self.use_centerness:
            nn.init.normal_(self.centerness_pred.weight, std=0.01)
            nn.init.zeros_(self.centerness_pred.bias)

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features: List of FPN outputs [P3, P4, P5], each [B, C, Hi, Wi]

        Returns:
            objectness:   List of [B, 1, Hi, Wi] per level (logits)
            bbox_reg:     List of [B, 4, Hi, Wi] per level (l, t, r, b)
            class_logits: List of [B, num_classes, Hi, Wi] per level, or None (class-agnostic)
            centerness:   List of [B, 1, Hi, Wi] per level, or None
        """
        all_objectness = []
        all_bbox_reg = []
        all_class_logits: List[torch.Tensor] = []
        all_centerness: List[torch.Tensor] = []

        for level_idx, feat in enumerate(features):
            # Objectness (shared tower for obj + class)
            obj_feat = self.obj_tower(feat)
            objectness = self.obj_pred(obj_feat)
            all_objectness.append(objectness)

            # Class logits (multi-class only)
            if self.multi_class:
                all_class_logits.append(self.class_pred(obj_feat))

            # BBox regression
            bbox_feat = self.bbox_tower(feat)
            bbox_reg = self.scales[level_idx](self.bbox_pred(bbox_feat))
            bbox_reg = torch.clamp(bbox_reg, max=4.0)  # Prevent exp() overflow
            bbox_reg = torch.exp(bbox_reg)
            all_bbox_reg.append(bbox_reg)

            # Centerness (optional)
            if self.use_centerness:
                centerness = self.centerness_pred(obj_feat)
                all_centerness.append(centerness)

        class_logits_out = all_class_logits if self.multi_class else None
        centerness_out = all_centerness if self.use_centerness else None
        return all_objectness, all_bbox_reg, class_logits_out, centerness_out


class _StandardConvBlock(nn.Module):
    """Standard Conv + BN + SiLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
