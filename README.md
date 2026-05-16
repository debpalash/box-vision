# BoxVision v2

**Lightweight, CPU-optimized, class-agnostic bounding box detector.**  
Purpose-built for workloads that only need to detect *where* objects are — no classification.

## Shipping Tiers

The right tier depends on dataset character. We trained the same recipes
on two datasets (shapes: small dense objects; road-signs: medium isolated
signs) and the conclusion is **different per dataset**. Numbers are
measured ONNX Runtime mAP@0.5 on Mac M1 CPU.

### Shapes lineup (567 train / 141 val)

| Tier | Backbone | params | input | mAP@0.5 | latency | use case |
|---|---|---|---|---|---|---|
| **Pro** | small + P2 | 500K | 416 | **87.30%** | 15.9ms | best accuracy |
| **Tiny+** ★ | tiny + P2 (KD, multi-scale, 300ep) | **164K** | **416** | **86.37%** | **8.5ms** | **best speed/accuracy** |
| Fast | small + P2 | 500K | 320 | 86.48% | 10.2ms | balanced |
| Small KD | small + P2 (KD, 300ep) | 842K | 416 | 78.02% | 17.5ms | (not recommended — noisier than Tiny+) |
| Tiny KD | tiny + P2 (KD) | 164K | 320 | 74.48% | 5.8ms | smallest @ 320 |
| Tiny KD INT8 | tiny + P2 (KD, PTQ) | 164K | 320 | 71.82% | 4.7ms | smallest deploy |

**On shapes, the sweet spot is Tiny+**: only 0.93 mAP below Pro at 1/3 the
params and 2× the speed. The 416 input + multi-scale training recipe
unlocked +11.9 mAP over the 320 baseline.

### Road-signs lineup (1376 train / 488 val)

| Tier | Backbone | params | input | mAP@0.5 | latency | use case |
|---|---|---|---|---|---|---|
| **Tiny KD** ★ | tiny + P2 (KD) | **164K** | **320** | **79.07%** | **4.1ms** | **best on this dataset** |
| Tiny KD INT8 | tiny + P2 (KD, PTQ) | 164K | 320 | 77.98% | 4.2ms | smallest deploy |
| Teacher | small + P2 (heavy aug) | 842K | 416 | 56.22% | — | underconfident at conf≥0.35 |
| Tiny+ (experimental) | tiny + P2 (KD, ms, 300ep) | 164K | 416 | 56.02% | 8.3ms | **regression — do not ship** |
| Small KD (experimental) | small + P2 (KD, 300ep) | 842K | 416 | 55.76% | 18.2ms | **regression — do not ship** |

**On road-signs, the shapes recipe regresses**: the 416+multi-scale upgrade
that lifted shapes by +12 mAP costs −23 mAP on road-signs. Likely cause:
the 4-way parallel pipeline used batch_size=32 without LR rescaling, and
road-signs is more sensitive to that than shapes. Until that's re-tested,
ship the **original Tiny KD @ 320** for road-signs.

### Visual scoring (20 panels, 10 per dataset)

We hand-scored a fresh 10-image sample from each val set:

- **Shapes**: Pro > Tiny+ > Fast >> Small KD > Tiny KD@320. Tiny+ matches
  Pro on most images, slightly noisier on the hardest (overlapping shapes
  + decorative spirals).
- **Road-signs**: Tiny KD@320 hits 9/10 visible signs vs. 7/10 for the
  experimental 416 variants and 5/10 for the teacher (the teacher's
  confidence calibration drops boxes below the conf=0.35 UI threshold).

> Shipping ONNX files (on disk):
> `boxvision-G.onnx` (Pro), `boxvision-G-320.onnx` (Fast),
> `boxvision-shapes-tiny-416-ms-300ep.onnx` (Tiny+ for shapes),
> `boxvision-tiny-p2.onnx` / `boxvision-rs-tiny-p2.onnx` (Tiny KD).

## Architecture

```
Input (320×320 default)
       │
       ▼
┌──────────────────────────┐
│  ShuffleNetV2 (0.5x/1.0x)│  ← No SE blocks (INT8-safe)
└──────────────────────────┘
       │
  ┌────┼────┐
  C3   C4   C5
  │    │    │
  ▼    ▼    ▼
┌──────────────────────────┐
│  Light FPN (top-down)     │  ← Ghost modules in 'small'
│  32/48ch unified          │  ← No bottom-up PAN path
└──────────────────────────┘
       │
  ┌────┼────┐
  P3   P4   P5
  │    │    │
  ▼    ▼    ▼
┌──────────────────────────┐
│  FCOS Head                │
│  • Objectness (1ch)       │
│  • BBox reg (4ch)         │
│  • No classification      │
│  • No centerness/DFL      │
└──────────────────────────┘
       │
       ▼
   NMS → Boxes
```

## Key Innovations (from research)

| Feature | Source | What it does |
|---|---|---|
| **Ghost-FPN** | NanoDet-Plus | ~2x param reduction in neck via cheap linear ops |
| **TAL + Soft Labels** | NanoDet DSLA | IoU-based soft targets instead of hard 0/1 (+2-3 mAP) |
| **EMA** | Standard | Smoothed weight averaging for better generalization |
| **ShuffleNetV2** | NanoDet | Quantization-safe backbone (no SE blocks) |

## Quick Start

```bash
uv sync
```

### Train

```bash
# Pro / Fast — small backbone with P2 (stride-4) head
uv run python -m boxvision.cli train \
    --preset small --use-p2 --input-size 416 \
    --dataset shapes --epochs 300

# Tiny — same idea, smaller backbone
uv run python -m boxvision.cli train \
    --preset tiny --use-p2 --input-size 320 \
    --dataset shapes --epochs 200
```

### Train Tiny via knowledge distillation (recommended for the Tiny tier)

```bash
# 1. Use a Pro/Fast checkpoint to label the training set with pseudo-boxes
uv run python scripts/distill_pseudo_labels.py \
    --teacher runs/pro/best.pt \
    --dataset shapes \
    --out datasets/shapes-distilled \
    --conf 0.40

# 2. Register the distilled dataset in datasets.yaml (see existing entries),
#    then train the tiny student with --use-p2 and lighter augmentation
uv run python scripts/train_kd_student.py \
    --tag tiny-kd-200ep --dataset shapes-distilled \
    --epochs 200 --use-p2 --light-aug
```

### Export to ONNX

```bash
uv run python -m boxvision.cli export \
    --checkpoint ./runs/best.pt \
    --output ./boxvision_model.onnx
```

### Detect + Benchmark

```bash
uv run python -m boxvision.cli detect \
    --model ./boxvision_model.onnx --image ./test.jpg

uv run python -m boxvision.cli benchmark \
    --model ./boxvision_model.onnx --image ./test.jpg
```

### Model Info

```bash
uv run python -m boxvision.cli info
```

## Dataset Format

COCO-format JSON. All categories collapsed to single "object" class.

```
data/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── train.json
    └── val.json
```

## Project Structure

```
boxvision/
├── __init__.py       # v0.2.0
├── config.py         # tiny/small presets + training config
├── backbone.py       # ShuffleNetV2 (0.5x / 1.0x)
├── fpn.py            # Ghost-FPN / standard FPN
├── head.py           # FCOS head (no centerness)
├── model.py          # Model assembly + EMA
├── losses.py         # Focal + GIoU + TAL (soft labels)
├── dataset.py        # COCO-format loader
├── train.py          # Training loop with EMA
├── evaluate.py       # mAP evaluation
├── export.py         # ONNX export + inference
├── inference.py      # PyTorch inference
└── cli.py            # CLI (train/export/detect/benchmark/info)
```

## Realistic Expectations

Numbers above are on shapes (567 train / 141 val). Measured mAP@0.5:0.95
on the same set: Pro 49.6%, Fast 49.2%, Tiny KD 34.6%, Tiny KD INT8 31.2%.

The speed/accuracy trade-off is intentional. We're not trying to match YOLO26n
accuracy — we're targeting the use case where box detection speed on CPU
matters more than a few mAP points. The Tiny tier is for edge/IoT workloads
that need millisecond-scale inference at the cost of ~13 mAP relative to Pro.

## Cross-Dataset Validation (road-signs)

To confirm the KD recipe transfers, we ran it end-to-end on Roboflow
road-signs (1376 train / 488 val, single class):

| Model | params | input | mAP@0.5 | latency |
|---|---|---|---|---|
| Teacher (small+P2+aug) | 842K | 416 | 56.22% | — |
| **Tiny KD FP32** | **164K** | **320** | **79.07%** | **4.14ms** |
| Tiny KD INT8 | 164K | 320 | 77.98% | 4.16ms |

The student decisively **outperforms its own teacher** here (+22.8 mAP). The
teacher's heavy mosaic+mixup+copy-paste recipe over-regularizes the
larger-object road-signs distribution, while the student's light-aug
recipe + the modest pseudo-label signal (~+18% extra boxes) lands in a
better basin. Worth keeping in mind: teacher hyperparams matter more than
absolute teacher mAP — a "good enough" teacher with sparse, high-precision
pseudo-labels can still drive a strong student.
