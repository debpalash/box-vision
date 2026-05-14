"""
Configuration for BoxVision v2.

Offers two presets:
- tiny:  ShuffleNetV2-0.5x, 32ch FPN, ~162K params — pure speed
- small: ShuffleNetV2-1.0x, 48ch FPN, ~500K params — balanced speed/accuracy
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # --- Backbone ---
    backbone: str = "shufflenet_v2_x1_0"  # shufflenet_v2_x0_5 or shufflenet_v2_x1_0
    pretrained_backbone: bool = True

    # --- FPN ---
    fpn_out_channels: int = 48          # 32 for tiny, 48 for small
    fpn_use_ghost: bool = True          # Use Ghost modules in FPN (from NanoDet)

    # --- Detection Head ---
    head_num_convs: int = 1
    head_use_depthwise: bool = True

    # --- Objectness ---
    objectness_threshold: float = 0.35
    nms_threshold: float = 0.50
    max_detections: int = 100

    # --- Input ---
    input_size: Tuple[int, int] = (320, 320)
    input_channels: int = 3

    # --- Stride levels ---
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])

    # --- Centerness ---
    use_centerness: bool = False


# Presets
def tiny_config(**overrides) -> ModelConfig:
    """Pure speed: 162K params, ~13ms CPU @ 320."""
    defaults = dict(
        backbone="shufflenet_v2_x0_5",
        fpn_out_channels=32,
        fpn_use_ghost=False,  # Not worth it at 32ch
        head_num_convs=1,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def small_config(**overrides) -> ModelConfig:
    """Balanced: ~500K params, ~15-18ms CPU @ 320. Targets ~25-27 mAP."""
    defaults = dict(
        backbone="shufflenet_v2_x1_0",
        fpn_out_channels=48,
        fpn_use_ghost=True,
        head_num_convs=2,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


@dataclass
class TrainConfig:
    """Training configuration."""

    # --- Dataset ---
    data_dir: str = "./data"
    annotation_format: str = "coco"
    train_ann: str = "train.json"
    val_ann: str = "val.json"

    # --- Training ---
    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # --- Scheduler ---
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    warmup_lr_ratio: float = 0.001

    # --- Loss weights ---
    loss_objectness_weight: float = 1.0
    loss_bbox_weight: float = 2.0

    # --- Focal loss params ---
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0

    # --- TAL assigner params ---
    tal_topk: int = 10
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    tal_use_soft_labels: bool = True  # DSLA-style soft IoU targets

    # --- Augmentation ---
    augment: bool = True
    mosaic: bool = True
    mosaic_off_epochs: int = 10

    # --- EMA ---
    use_ema: bool = True
    ema_decay: float = 0.9999

    # --- Checkpointing ---
    save_dir: str = "./runs"
    save_interval: int = 5
    eval_interval: int = 5

    # --- Device ---
    device: str = "cpu"

    # --- Mixed precision ---
    amp: bool = False


@dataclass
class ExportConfig:
    """ONNX export configuration."""

    output_path: str = "./boxvision_model.onnx"
    opset_version: int = 13
    dynamic_axes: bool = True
    simplify: bool = True
    quantize_int8: bool = False
    input_size: Tuple[int, int] = (320, 320)
