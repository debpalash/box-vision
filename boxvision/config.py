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

    # --- Multi-class ---
    # num_classes = 1 → class-agnostic (objectness only, the original BoxVision)
    # num_classes > 1 → multi-class detection (adds a class_pred head)
    num_classes: int = 1

    # --- Objectness / postprocess ---
    # nms_threshold = 0.5 keeps the "near-miss" boxes that contribute to
    # mAP@0.5:0.95. For visual cleanliness ("one box per object"), enable WBF
    # (fuse clusters via score-weighted average) — gives clean output and
    # also helps mAP since fused boxes are usually better-localized.
    objectness_threshold: float = 0.05
    nms_threshold: float = 0.50
    containment_threshold: float = 0.0   # 0 = off; raise (0.4-0.6) for clean viz
    use_wbf: bool = False                 # if True, replace NMS with WBF
    wbf_iou_threshold: float = 0.55       # cluster boxes whose IoU exceeds this
    max_detections: int = 100

    # --- Input ---
    input_size: Tuple[int, int] = (320, 320)
    input_channels: int = 3

    # --- Stride levels ---
    # When use_p2=True, the backbone exposes its stem (stride 4) and the FPN
    # gains a P2 level. Pushes small-object recall up substantially at a
    # modest param/latency cost.
    use_p2: bool = False
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
        fpn_use_ghost=False,
        head_num_convs=2,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


@dataclass
class TrainConfig:
    """Training configuration."""

    # --- Dataset ---
    # Dataset is referenced by registry name (see datasets.yaml).
    # Paths and format are resolved via boxvision.registry.load_dataset().
    dataset: str = "road-signs"

    # --- Training ---
    epochs: int = 100
    # Fixed seed for weight init, shuffling, and augmentation sampling. The
    # autoresearch series runs every experiment at the same seed so metric
    # deltas reflect the code change, not sampling variance (unseeded
    # identical-code runs differed by 1.88 mAP50). -1 disables seeding.
    seed: int = 42
    # Wall-clock training budget in minutes; 0 disables. When set, the epoch
    # loop stops at the deadline (autoresearch fixed-budget contract) after a
    # final validation + checkpoint save.
    max_minutes: float = 0.0
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
    loss_objectness_weight: float = 4.0
    loss_bbox_weight: float = 6.0
    loss_class_weight: float = 1.0  # Multi-class only

    # --- Focal loss params ---
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0

    # --- TAL assigner params ---
    # Stays at 10. Lowering to 5 hurt strict-IoU mAP on small datasets
    # (insufficient positive samples → noisier bbox regression).
    tal_topk: int = 10
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    tal_use_soft_labels: bool = True  # DSLA-style soft IoU targets

    # --- Augmentation ---
    augment: bool = True
    mosaic: bool = True
    mosaic_off_epochs: int = 10
    # Multi-scale training: when enabled, pick a random input size from
    # `multiscale_sizes` at the start of each epoch. Trains scale invariance.
    multiscale: bool = False
    multiscale_sizes: List[int] = field(default_factory=lambda: [320, 416, 512])
    mixup: bool = False
    copy_paste: bool = False

    # --- EMA ---
    # Tuned for small datasets (1-2K images): 0.999 takes ~5K steps to converge,
    # which means EMA is mostly init noise for the first 30+ epochs on small data.
    # 0.99 converges within ~500 steps. Drop EMA entirely if training <500 total steps.
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_warmup_steps: int = 50  # Skip EMA updates until raw model has moved meaningfully

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
