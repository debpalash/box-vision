# BoxVision v2

**Lightweight, CPU-optimized, class-agnostic bounding box detector.**  
Purpose-built for workloads that only need to detect *where* objects are — no classification.

## Three Shipping Tiers

Measured on shapes (567-image dataset, 141-image val), Mac M1 CPU, ONNX Runtime:

| Tier | Backbone | params | input | mAP@0.5 | latency | use case |
|---|---|---|---|---|---|---|
| **Pro** | small + P2 | 500K | 416 | **87.30%** | 15.9ms | best accuracy |
| **Fast** | small + P2 | 500K | 320 | 86.48% | 10.2ms | balanced |
| **Tiny KD** | tiny + P2 (distilled) | **164K** | 320 | 74.48% | **5.8ms** | edge/IoT speed |
| **Tiny KD INT8** | tiny + P2 (distilled, PTQ) | 164K | 320 | 71.82% | **4.7ms** | smallest deploy |

The Tiny tier is the output of offline knowledge distillation: a `small+P2`
teacher labels the training set, and the student trains on real GT + teacher
pseudo-labels (~+48% extra training signal). The recipe gives Tiny **86% of
the teacher's mAP at 33% of the parameters and 1.8× the speed**.

> Tier names map to shipping ONNX files: `boxvision-G.onnx` (Pro),
> `boxvision-G-320.onnx` (Fast), `boxvision-tiny-p2.onnx` (Tiny),
> `boxvision-tiny-p2-int8.onnx` (Tiny INT8).

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
