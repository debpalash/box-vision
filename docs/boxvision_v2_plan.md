# BoxVision v2 — Redesign Plan

## The Target to Beat

**YOLO26n** (our benchmark):
| Metric | YOLO26n |
|---|---|
| Params | 2.57M |
| FLOPs | 6.1 GFLOPs |
| mAP@50:95 | ~40.9 (80 classes) |
| CPU (ONNX) | ~38.9ms |
| NMS | None (end-to-end) |

**Our goal**: Surpass YOLO26n CPU inference by **multiple folds** (target: <10ms ONNX, <15ms worst case).

## Why We Can Win on Speed

YOLO26n carries overhead we don't need:

1. **80-class classification head** — we need 0 classes (objectness only)
   - YOLO26n's Detect head outputs `(N, 84, 8400)` = 80 class logits + 4 bbox per anchor
   - We output `(N, 5, K)` = 1 objectness + 4 bbox — **16x fewer output channels**
2. **2.57M params / 6.1 GFLOPs** — most budget goes to multi-class feature richness
   - Class-agnostic needs far less feature diversity
3. **C2PSA attention module** — expensive, designed for class discrimination
4. **Full PAN-FPN bidirectional neck** — 3 upsample + 3 downsample paths
5. **reg_max=1 but still carries DFL-era channel overhead** in the head

## BoxVision v2 Architecture

### Design Principles
- **Extreme efficiency**: Target <500K params, <1 GFLOP
- **320×320 input** as default (640 optional) — 4x less compute than YOLO26
- **Single-pass neck** (no bidirectional PAN — top-down FPN only)
- **NMS-free end-to-end** (one-to-one matching like YOLO26)
- **INT8-first design**: all ops quantization-friendly (no LayerNorm, no SE blocks)
- **No classification head**: pure objectness + bbox

### Architecture

```
Input (320×320 or 640×640)
       │
       ▼
┌─────────────────────────┐
│  MobileNetV4-Conv-S     │  ← 2024 backbone, faster than MNv3
│  OR ShuffleNetV2-0.5x   │  ← Even lighter alternative
│  (quantization-friendly) │
└─────────────────────────┘
       │
  ┌────┼────┐
  C3   C4   C5               ← stride 8/16/32
  │    │    │
  ▼    ▼    ▼
┌─────────────────────────┐
│  Slim FPN (top-down only)│  ← No bottom-up path (saves ~40% neck FLOPs)
│  32 channels unified     │  ← Half of v1's 64ch
│  Depthwise-sep convs     │
└─────────────────────────┘
       │
  ┌────┼────┐
  P3   P4   P5
  │    │    │
  ▼    ▼    ▼
┌─────────────────────────┐
│  Micro Head              │
│  • 1 DWConv per branch   │  ← Minimal: 1 conv, not 2-4
│  • Objectness (1ch)      │
│  • BBox reg (4ch)        │
│  • No centerness         │  ← Removed (YOLO26 proved it's not needed)
│  • No DFL                │  ← Aligned with YOLO26
└─────────────────────────┘
       │
       ▼
  End-to-end (one-to-one)    ← NMS-free at inference
  OR simple top-K + NMS      ← Fallback for ONNX export simplicity
```

### Speed Budget Analysis (320×320 input)

| Component | Est. FLOPs | Est. Params | Notes |
|---|---|---|---|
| Backbone (ShuffleNetV2-0.5x) | ~0.15G | ~350K | Lightest viable backbone |
| FPN (32ch, top-down only) | ~0.02G | ~15K | 3 lateral + 3 smooth convs |
| Head (1 DWConv + pred) | ~0.01G | ~5K | Minimal |
| **Total** | **~0.18G** | **~370K** | **34x fewer FLOPs than YOLO26n** |

With 640×640:
| Component | Est. FLOPs | Est. Params |
|---|---|---|
| **Total @ 640** | **~0.72G** | **~370K** |
| vs YOLO26n @ 640 | 6.1G | 2.57M |
| **Ratio** | **8.5x fewer** | **7x fewer** |

### Expected CPU Latency

| Config | Est. ONNX CPU (ms) | vs YOLO26n |
|---|---|---|
| BoxVision v2 @ 320, FP32 | ~5-8ms | **5-8x faster** |
| BoxVision v2 @ 320, INT8 | ~2-4ms | **10-20x faster** |
| BoxVision v2 @ 640, FP32 | ~15-20ms | **2-3x faster** |
| BoxVision v2 @ 640, INT8 | ~5-8ms | **5-8x faster** |
| YOLO26n @ 640, ONNX | ~38.9ms | baseline |

### Training Strategy

1. **Progressive Loss (ProgLoss)** — from YOLO26, ramp loss weights during training
2. **Task-Aligned Assigner (STAL/TAL)** — from YOLO26, better positive sample selection  
3. **EMA** — exponential moving average of weights
4. **Mosaic augmentation** — ON for 80% of training, OFF for last 20%
5. **MuSGD optimizer** — adopt YOLO26's hybrid optimizer
6. **Knowledge distillation** — use YOLO26m as teacher, supervise box outputs
7. **Train on COCO** with all 80 classes collapsed to single "object" class

### Accuracy Expectations (Honest)

On class-agnostic COCO eval:
- **BoxVision v2 @ 640**: ~30-35 mAP@50:95 (class-agnostic)
- **YOLO26n @ 640 (1-class retrain)**: ~35-38 mAP@50:95
- **Gap**: ~3-5 mAP points — acceptable given **5-8x speed advantage**

The trade-off: we're not trying to match YOLO26n on accuracy. We're trying to be **accurate enough** while being **multiple folds faster** on CPU.

### Backbone Options (Ranked)

| Backbone | Params | FLOPs@224 | Top-1 ImageNet | Quant-friendly |
|---|---|---|---|---|
| ShuffleNetV2-0.5x | 350K | 0.04G | 60.3 | ✓✓✓ |
| MobileNetV4-Conv-S | ~3.5M | 0.2G | 73.8 | ✓✓ |
| GhostNetV2-0.5x | ~2.5M | 0.04G | 66.7 | ✓✓ |
| MobileNetV3-Small (current) | 2.5M | 0.06G | 67.7 | ✓ (SE blocks hurt INT8) |

**Recommendation**: Start with **ShuffleNetV2-0.5x** for maximum speed. If accuracy insufficient, step up to GhostNetV2-0.5x.

### Implementation Phases

1. **Phase 1**: Swap backbone to ShuffleNetV2, slim down FPN to 32ch, strip head to minimal
2. **Phase 2**: Add TAL assigner, EMA, mosaic
3. **Phase 3**: ONNX export + INT8 quantization + benchmarking vs YOLO26n
4. **Phase 4**: Knowledge distillation from YOLO26m teacher (if accuracy gap too large)

## Files to Change

| File | Changes |
|---|---|
| `backbone.py` | Replace MobileNetV3 → ShuffleNetV2-0.5x |
| `fpn.py` | Reduce to 32ch, remove bottom-up path |
| `head.py` | Strip to 1 DWConv per branch, remove centerness |
| `model.py` | Update assembly, add end-to-end option |
| `losses.py` | Add TAL assigner, remove centerness loss |
| `train.py` | Add EMA, mosaic scheduling, ProgLoss |
| `config.py` | Update defaults (320 input, 32ch FPN, etc.) |
| `export.py` | Ensure clean INT8 quantization path |
