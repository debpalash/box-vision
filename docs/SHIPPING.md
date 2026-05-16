# BoxVision — Shipping Notes

A single document covering: what we shipped, how to use it in production,
and where to push next for higher accuracy.

---

## 1. What's shipped

BoxVision is a class-agnostic FCOS-style detector built for **CPU
inference** on small/medium datasets. The codebase ships four working
tiers, validated on two datasets:

- **shapes** — 567 train / 141 val, small dense objects (Label Studio export)
- **road-signs** — 1376 train / 488 val, medium isolated signs (Roboflow)

All numbers below are ONNX Runtime mAP@0.5 on Mac M1 CPU. Latency is a
warm 5-run average per image at the model's native input size.

### Tier lineup

| Tier | Backbone | Params | Input | shapes mAP | road-signs mAP | Latency |
|---|---|---|---|---|---|---|
| **Pro** (G) | small + P2 | 500K | 416 | **87.30%** | n/a | 15.9 ms |
| **Fast** (G) | small + P2 | 500K | 320 | 86.48% | n/a | 10.2 ms |
| **Tiny+** ★ shapes | tiny + P2, KD, multi-scale, 300 ep | **164K** | 416 | **86.37%** | (regression, do not ship) | **8.5 ms** |
| **Tiny KD** ★ road-signs | tiny + P2, KD | 164K | 320 | 74.48% | **79.07%** | 4.1 ms |
| Tiny KD INT8 | tiny + P2, KD, static PTQ | 164K | 320 | 71.82% | 77.98% | 4.7 ms |

★ = the dataset-specific shipping recommendation.

### Where things live

```
runs/
  G-aug-300ep/                       # Pro + Fast teacher (shapes)
    boxvision-G.onnx                 # 416 input
    boxvision-G-320.onnx             # 320 input
    boxvision-G-320-int8.onnx        # 320 INT8 PTQ
  shapes-tiny-416-ms-300ep/          # Tiny+ shapes student
    boxvision-shapes-tiny-416-ms-300ep.onnx
  tiny-p2-light-200ep/               # Tiny KD shapes student @ 320
    boxvision-tiny-p2.onnx
    boxvision-tiny-p2-int8.onnx
  rs-tiny-kd-200ep/                  # Tiny KD road-signs student @ 320
    boxvision-rs-tiny-p2.onnx
    boxvision-rs-tiny-p2-int8.onnx
```

### Key engineering wins

1. **Offline knowledge distillation works**: training the small Tiny
   student on real GT + teacher pseudo-labels closes the accuracy gap to
   the teacher (shapes: 86% of teacher mAP at 33% of params).
2. **Per-channel INT8 PTQ is essential** for depthwise convs — naive
   per-tensor quantization tanked Tiny mAP from 74% to 39% on shapes.
   Per-channel restored it to 71.82%.
3. **Recipe ≠ universal**. The 416-input + multi-scale + 300-epoch recipe
   that gave Tiny+ a +12 mAP boost on shapes regressed road-signs by −23
   mAP. Always validate per dataset.

### Known limitations (honest)

- **Datasets are small** (708 + 2093 images total). Apples-to-apples vs
  YOLO26n on COCO is not yet measured.
- **mAP@0.5:0.95 is mediocre** (33-48% across tiers). Localization at
  strict IoU is weaker than mAP@0.5 implies. This is a known FCOS-without-DFL
  symptom.
- **Teacher confidence calibration**: heavy-aug teachers output low scores
  at inference. The 56% road-signs teacher mAP is correct but its boxes
  fall below `conf=0.35` in visual galleries.
- **No real-device latency**. Numbers are Mac M1 CPU ONNX Runtime,
  4 threads. Embedded/x86 numbers untested.

---

## 2. Production usage

### Install

```bash
uv sync                              # or: pip install -r requirements.txt
```

`requirements`: pytorch, torchvision, onnx, onnxruntime, opencv-python,
numpy, pycocotools, pyyaml.

### Pick the right ONNX model

```python
# Class-agnostic object detection — pick by dataset character.
MODEL = "runs/shapes-tiny-416-ms-300ep/boxvision-shapes-tiny-416-ms-300ep.onnx"
INPUT = 416                          # must match the model's training resolution

# For road-signs–style content (medium isolated objects):
# MODEL, INPUT = "runs/rs-tiny-kd-200ep/boxvision-rs-tiny-p2.onnx", 320
```

### Inference (minimal)

```python
import cv2, numpy as np, onnxruntime as ort, torch, torchvision.ops as ops

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])

def letterbox(img, size):
    H, W = img.shape[:2]
    s = min(size / H, size / W)
    nh, nw = int(round(H * s)), int(round(W * s))
    r = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = r
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return rgb.transpose(2, 0, 1).astype(np.float32), s

def predict(img_bgr, conf=0.35, iou=0.5):
    tensor, scale = letterbox(img_bgr, INPUT)
    boxes, scores, labels = sess.run(None, {"input": tensor[None, ...]})
    keep = scores > conf
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if len(boxes) > 0:
        idx = ops.batched_nms(torch.from_numpy(boxes),
                              torch.from_numpy(scores),
                              torch.from_numpy(labels),
                              iou).numpy()[:300]
        boxes, scores = boxes[idx] / scale, scores[idx]
    return boxes, scores

img = cv2.imread("path/to/image.jpg")
boxes, scores = predict(img)
```

### Confidence/NMS tuning

- `conf=0.05` for full recall (mAP eval default)
- `conf=0.35` for clean visualization
- `iou=0.5` standard NMS
- For "one box per object" mode, drop `iou` to ~0.15 and add containment
  suppression (see `scripts/test_inference.py` for reference)

### Train your own variant

```bash
# 1. Register your dataset in datasets.yaml (see existing entries)

# 2. Train a Pro/Fast teacher
python -m boxvision.cli train --preset small --use-p2 \
    --input-size 416 --dataset YOUR_DATASET --epochs 300

# 3. Pseudo-label your training set with the teacher
python scripts/distill_pseudo_labels.py \
    --teacher runs/pro/best.pt \
    --dataset YOUR_DATASET \
    --out datasets/YOUR_DATASET-distilled \
    --conf 0.40

# 4. Train the Tiny student
python scripts/train_kd_student.py --tag tiny-kd-200ep \
    --dataset YOUR_DATASET-distilled \
    --preset tiny --use-p2 --light-aug \
    --epochs 200

# 5. Export
python -m boxvision.cli export \
    --checkpoint runs/tiny-kd-200ep/best.pt \
    --output runs/tiny-kd-200ep/model.onnx

# 6. (Optional) INT8 PTQ
python scripts/static_int8_ptq.py \
    --onnx runs/tiny-kd-200ep/model.onnx \
    --dataset YOUR_DATASET --input-size 320 \
    --out runs/tiny-kd-200ep/model-int8.onnx
# Note: edit static_int8_ptq.py to set per_channel=True before running.
```

### Evaluation

```bash
# Single ONNX model on a registered dataset's val split
python scripts/eval_onnx.py --onnx PATH.onnx \
    --dataset YOUR_DATASET --input-size 320 \
    --conf 0.05 --nms-iou 0.50

# Side-by-side visual gallery vs the Pro teacher
python scripts/test_inference.py --dataset shapes \
    --n-images 10 --out runs/gallery/ --conf 0.35
```

### Remote training (RTX 4090)

The `boxship` CLI ships code to the GPU box and streams training output.
See `docs/boxvision_v2_plan.md` for setup, and `.boxship.env` template.

```bash
./boxship/boxship inspect
./boxship/boxship upload scripts/ box-vision/scripts/
./boxship/boxship run "cd ~/box-vision && python scripts/train_kd_student.py ..."
./boxship/boxship wait ~/box-vision/runs/X.log --marker='Training complete'
```

---

## 3. Where to push for higher accuracy

Ordered by expected return. Items marked **[R]** are research-grade
(falsifiable hypothesis); **[E]** are engineering wins.

### Tier A — likely +3-5 mAP each, modest effort

1. **[E] Resolve the road-signs regression.** The 416/multi-scale recipe
   gained shapes +12 but lost road-signs −23. Strongest hypothesis: the
   parallel-pipeline `batch_size=32` (vs. baseline 64) without LR
   rescaling. Re-run road-signs Tiny+ with bs=64 (or bs=32 with `lr=0.005`)
   to confirm or refute. Cheap A/B; high information value.
2. **[E] DFL (Distribution Focal Loss).** Replace direct bbox regression
   with a discretized distribution head. Standard YOLO improvement worth
   +1-2 mAP@0.5 and +3-5 mAP@0.5:0.95 (where we're weakest).
3. **[E] Centerness head.** Already gated in `config.py`
   (`use_centerness: bool = False`). Re-enable, train, measure. Cheap
   and reversible.
4. **[E] Teacher-strength sweep.** The road-signs teacher capped at 56%
   despite being a "Pro" recipe; the shapes teacher hits 87%. Either the
   road-signs recipe wasn't actually Pro-equivalent, or the dataset's
   ceiling is lower. Train a properly-tuned road-signs teacher (small +
   P2 + heavy-aug + 416 + multi-scale + 500 ep + bs=64) and re-distill.
   If teacher hits 70%+, student likely hits 85%+.
5. **[E] TTA at inference.** `tta.py` exists, untested in benchmark.
   Horizontal-flip TTA usually adds +0.5-1.5 mAP at 2× latency. Real cost
   for production but useful for offline scoring.

### Tier B — possibly +5-10 mAP, bigger effort

6. **[R] PrototypeDet** *(the headline research bet from `goal.md`)*.
   Replace the FCOS classifier head with metric-learning prototypes
   bootstrapped from SAM masks. Per-spatial-location cosine similarity
   against K learned prototypes instead of a learned binary classifier.
   Plausibly +3-8 mAP at the tiny scale; falsifiable; publishable either
   way. **Untouched as of this session.**
7. **[R] Online distillation (response-level).** Current KD is offline:
   teacher emits a few extra bboxes once. Online KD streams teacher
   logits per batch and applies a soft-label loss to the student's
   pre-NMS objectness map. Expected +2-4 mAP@0.5:0.95 over offline KD.
8. **[E] Bigger backbone family.** ShuffleNetV2-1.5x at 720K params sits
   between Tiny and Small. Drop-in via `backbone="shufflenet_v2_x1_5"`.
   Worth a single training run to map the params-vs-mAP curve.
9. **[E] Multi-scale + DFL + centerness combined recipe** on a properly
   tuned learning rate. Run the same 4-variant sweep with bs=64, lr
   rescaled, and either DFL or centerness on/off.

### Tier C — open-ended / data side

10. **[E] More data, not more recipe.** The shapes/road-signs ceilings
    look real for these sizes. We confirmed in this session that
    duplicating road-signs data (1376 → 1884) didn't help. To push past
    79%, get a 5-10× larger annotated set, or generate one via
    SAM-on-unlabeled-images + manual curation.
11. **[R] SAM-bootstrapped pretraining.** Pre-train the backbone on
    bbox-mask pairs derived from SAM run over an unlabeled image pool
    (e.g. SA-1B subset, see `datasets.yaml` future entry). Even modest
    pretraining usually transfers +3-5 mAP on downstream small datasets.
12. **[E] Domain-specific augmentation.** Mosaic + mixup + copy-paste is
    YOLO heritage. For sign detection, we suspect simpler aug (just
    flips + photometric) may work better — testable via single A/B.
13. **[E] Apples-to-apples vs YOLO26n.** Run both ONNX models on the same
    box (the remote x86 64-core machine is available). Without this, all
    "vs YOLO26n" claims have an asterisk.

### Tier D — production hardening

14. **[E] INT8 latency story on x86 / ARM.** Mac M1 shows ~no win for INT8
    on Tiny (QDQ overhead). Likely a real win on x86 AVX2/AVX-512 and
    on mobile. Untested. Run `scripts/static_int8_ptq.py` benchmark on
    the actual deployment target.
15. **[E] Model-card per variant**. README has tier numbers but no
    error-mode docs. For each shipped variant: document failure modes
    (occluded objects, small + decorative overlays for shapes,
    underconfidence at high conf threshold for the teacher).
16. **[E] Streaming/video pipeline.** Current inference is per-image.
    For video, add tracking (ByteTrack or simple IoU-based) + temporal
    confidence smoothing.

### Order I'd actually take

If we resume tomorrow:

1. **First**: re-run road-signs Tiny+ with bs=64 (Tier A #1). Either
   recovers the recipe or kills the multi-scale hypothesis. ~30 min.
2. **Then**: enable centerness + DFL together, retrain Tiny+ on shapes
   (Tier A #2-3). ~30 min remote.
3. **Then**: write PrototypeDet PoC (Tier B #6). Multi-day. This is the
   actual research bet; everything else is engineering.

The first two pay for themselves in raw mAP. The third is where
BoxVision becomes interesting beyond "tiny FCOS with offline KD".
