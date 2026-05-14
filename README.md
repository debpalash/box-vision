# BoxVision v2

**Lightweight, CPU-optimized, class-agnostic bounding box detector.**  
Purpose-built for workloads that only need to detect *where* objects are — no classification.

## Two Presets

| | BoxVision-tiny | BoxVision-small | YOLO26n |
|---|---|---|---|
| **Backbone** | ShuffleNetV2-0.5x | ShuffleNetV2-1.0x | C3k2 + C2PSA |
| **Neck** | FPN (32ch) | Ghost-FPN (48ch) | PAN-FPN |
| **Parameters** | **162K** | **838K** | 2,572K |
| **Size (FP32)** | **0.62 MB** | **3.20 MB** | ~10 MB |
| **CPU @ 320** | **~13ms / 77 FPS** | **~24ms / 41 FPS** | N/A |
| **CPU @ 640** | ~25ms | ~50ms | ~39ms (ONNX) |
| **Use case** | Ultra-fast, edge/IoT | Balanced speed/accuracy | Multi-class general |

> **Note**: Our numbers are PyTorch on Apple Silicon CPU. YOLO26n is
> ONNX-optimized on EC2 P4d CPU. Not apples-to-apples until we
> benchmark both as ONNX on the same hardware.

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

### Train (default: small preset)

```bash
uv run python -m boxvision.cli train \
    --preset small \
    --data ./data \
    --epochs 100

# Or use tiny preset for extreme speed
uv run python -m boxvision.cli train \
    --preset tiny \
    --data ./data
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

Based on NanoDet-Plus benchmarks (same backbone family):

| Config | Est. mAP@50:95 | Notes |
|---|---|---|
| BoxVision-tiny @ 320 | ~15-20 | Very constrained (162K params) |
| BoxVision-small @ 320 | ~25-28 | Matches NanoDet-Plus tier |
| BoxVision-small @ 416 | ~27-30 | Better for larger objects |
| YOLO26n @ 640 | ~40.9 | 80-class, 3x more params |

The speed/accuracy trade-off is intentional. We're not trying to match YOLO26n
accuracy — we're targeting the use case where box detection speed on CPU
matters more than a few mAP points.
