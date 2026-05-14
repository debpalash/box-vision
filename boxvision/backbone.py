"""
Backbone v2: ShuffleNetV2 feature extractor.

Supports two variants:
- 0.5x: 350K params, 48/96/192 channels (tiny)
- 1.0x: 1.26M params, 116/232/464 channels (small)

Why ShuffleNetV2 over MobileNetV3:
- No SE blocks → clean INT8 quantization
- Channel shuffle is free on CPU (reshape)
- Proven backbone in NanoDet-Plus (27 mAP at 320px with 1.0x)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Output channels per variant (after each stage: stage2, stage3, stage4)
_VARIANT_CHANNELS = {
    "shufflenet_v2_x0_5": [48, 96, 192],
    "shufflenet_v2_x1_0": [116, 232, 464],
}

_VARIANT_WEIGHTS = {
    "shufflenet_v2_x0_5": models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1,
    "shufflenet_v2_x1_0": models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
}

_VARIANT_FACTORY = {
    "shufflenet_v2_x0_5": models.shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
}


class ShuffleNetV2Backbone(nn.Module):
    """
    ShuffleNetV2 backbone for multi-scale feature extraction.

    Extracts features at 3 scales (strides 8, 16, 32).

    Output channels:
        0.5x: C3=48,  C4=96,  C5=192
        1.0x: C3=116, C4=232, C5=464
    """

    def __init__(self, variant: str = "shufflenet_v2_x1_0", pretrained: bool = True):
        super().__init__()

        if variant not in _VARIANT_FACTORY:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(_VARIANT_FACTORY.keys())}")

        weights = _VARIANT_WEIGHTS[variant] if pretrained else None
        shufflenet = _VARIANT_FACTORY[variant](weights=weights)

        # Stage 0: conv1 + maxpool → stride 4
        self.stem = nn.Sequential(
            shufflenet.conv1,
            shufflenet.maxpool,
        )

        # Stage 1 (torchvision stage2): stride 4→8
        self.stage1 = shufflenet.stage2  # → C3

        # Stage 2 (torchvision stage3): stride 8→16
        self.stage2 = shufflenet.stage3  # → C4

        # Stage 3 (torchvision stage4): stride 16→32
        self.stage3 = shufflenet.stage4  # → C5
        # Skip conv5 (1024ch 1x1 conv) — unnecessary for detection

        self.out_channels_list = list(_VARIANT_CHANNELS[variant])
        self.variant = variant

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            List of feature maps: [C3, C4, C5]
        """
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]

    def get_out_channels(self):
        """Return output channels for each feature level."""
        return self.out_channels_list


def _verify_backbone_output():
    """Sanity check for both backbone variants."""
    for variant in ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0"]:
        backbone = ShuffleNetV2Backbone(variant=variant, pretrained=False)
        x = torch.randn(1, 3, 320, 320)
        features = backbone(x)

        total_params = sum(p.numel() for p in backbone.parameters())
        channels = _VARIANT_CHANNELS[variant]

        print(f"\n{variant}")
        print(f"  Params: {total_params:,}")
        for i, feat in enumerate(features):
            expected_ch = channels[i]
            assert feat.shape[1] == expected_ch, f"Channel mismatch at C{i+3}"
            print(f"  C{i+3}: {list(feat.shape)} (stride {2**(i+3)})")

    print("\nAll backbone variants verified!")


if __name__ == "__main__":
    _verify_backbone_output()
