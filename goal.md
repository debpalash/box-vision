# BoxVision — Lightweight CPU Box Detector

## Goal

A class-agnostic bounding box detector optimized for CPU inference.
Target: **multiple-fold CPU speedup vs YOLO26n** at deployable accuracy on single-class (objectness) detection.

**Differentiating bet**: **PrototypeDet** — replace the classifier objectness head with metric-learning prototypes bootstrapped from SAM masks. A class-agnostic supervision signal multi-class models structurally can't use. See [Research Direction](#research-direction--prototypedet-the-ambitious-bet).

Reference points:

- **YOLO26n** (Jan 2026): 2.57M params, 40.9 mAP@50:95 on COCO, 38.9ms CPU ONNX @ 640
- **NanoDet-Plus-m** (2022): 1.17M params, 27.0 mAP @ 320, **5.25ms on i7-8700**
- **Our v2 target**: 162K params, ~25-30 mAP@0.5, **<10ms CPU** @ 320 (ONNX, single-thread)

---

## v2 — Current State (Implemented & Verified)

### Architecture

| Component | Choice | Params |
| --- | --- | --- |
| Backbone | ShuffleNetV2-0.5x (no SE, INT8-safe, skip conv5) | 143K |
| Neck | Top-down-only FPN, 32 channels, depthwise-separable smoothing | 15K |
| Head | 1-conv FCOS, class-agnostic, no centerness, no DFL | 4.2K |
| Assigner | Task-Aligned Assigner (TAL), topk=10, α=0.5, β=6.0 | — |
| Loss | Focal (objectness) + GIoU (bbox) | — |
| Training | SGD + cosine warmup, EMA (decay 0.9999), AMP optional | — |
| **Total** | — | **162,312 params · 0.62 MB FP32** |

### Verified Measurements

| Metric | Value | Method |
| --- | --- | --- |
| Params | 162,312 | `model.count_parameters()` |
| FP32 size | 0.62 MB | computed |
| Latency @ 320×320 | **13.7ms / 73 FPS** | PyTorch eager, 4 threads, 30-run avg |
| Latency @ 640×640 | **26.6ms / 38 FPS** | PyTorch eager, 4 threads, 30-run avg |

ONNX export latency unverified — expected ~50-70% of PyTorch eager.

### Honest Gaps in v2 (Found in Review)

1. **Mosaic augmentation is NOT implemented.** Config flag `mosaic: True` exists; [dataset.py](boxvision/dataset.py) has no mosaic code. `mosaic_off_epochs` field is read by nothing.
2. **`torch.exp(bbox_reg)` in [head.py:119](boxvision/head.py#L119) has no clamp** — risk of `inf`/`NaN` at training start. Need `torch.exp(bbox_reg.clamp(max=8))`.
3. **Decode + NMS lives in `forward()` with dynamic `torch.arange`** — ONNX export likely to fail or produce slow graph. Must refactor to export raw logits, run decode in Python wrapper.
4. **Head capacity is razor-thin (4.2K params)** — likely accuracy bottleneck. Worth 2-3× more (~10K).
5. **INT8 QAT not implemented** — `quantize_int8` config flag exists, no QAT loop in trainer.

---

## Research Direction — PrototypeDet (the ambitious bet)

The core architectural innovation. Replaces the standard "objectness classifier head" with **prototype-based metric learning**, bootstrapped from SAM masks. Plausibly paper-worthy if it works.

### The core idea

Every tiny detector (NanoDet, PicoDet, YOLO-nano, our v2) uses a per-spatial-location classifier head for objectness — a CNN that outputs a logit per grid cell. We replace it with:

```python
projected_feat = 1x1Conv(F)              # B, D, H, W   — project features to embedding space
prototypes     = learned_param           # K, D         — K learned "object" prototypes
similarity     = cosine_sim(projected_feat, prototypes)  # B, K, H, W
objectness     = temperature * max_k(similarity)         # B, 1, H, W
```

The model no longer asks "is this cell a positive sample?" via a learned classifier. It asks **"does this cell's feature embedding look like one of K learned object prototypes?"** via metric learning.

### Why this is research-grade (not just engineering)

1. **Novel architecture for tiny detection.** Metric-learning heads exist in classification (CLIP, ProtoNet) and segmentation (Mask2Former queries). No published tiny detector uses prototype-based objectness. DETR uses ~100 object queries via a decoder; we use K=4 prototypes with cosine similarity at every spatial location — different mechanism entirely.

2. **Class-agnostic is the natural fit.** "What is an object?" reduces to "what's in the learned prototype set?" Multi-class detectors structurally cannot use this framing because they need per-class classifiers.

3. **Bootstraps perfectly from SAM.** SAM masks provide positive/negative patch pairs for free. Pretrain prototypes via InfoNCE on SAM mask embeddings vs background. Self-supervised on millions of images.

4. **Different generalization story.** A classifier head memorizes COCO statistics. Learned prototypes are forced to extract a *visual essence of objectness* that hypothetically transfers across domains (road signs, medical, aerial) more cleanly. Falsifiable hypothesis we can test.

5. **Falsifiable = publishable either way.** Either prototype-based heads match classifier heads at tiny scales, or they don't. Either they transfer better cross-domain, or they don't. Both outcomes are publishable.

### Architecture diff vs v2

| Component | v2 | PrototypeDet |
| --- | --- | --- |
| Backbone | ShuffleNetV2-0.5x (143K) | unchanged |
| Neck | top-down FPN 32ch (15K) | unchanged |
| **Objectness head** | 1 conv tower + 1 pred conv = **4K params, classifier-style** | **1×1 projection (32→128) + K=4 prototypes (4×128) = 4.6K params, prototype-style** |
| bbox head | 1 conv tower + 1 pred conv = ~3K | unchanged |
| Loss | Focal on objectness | **InfoNCE pretrain → BCE on similarity logits** |
| Pretraining | ImageNet (backbone only) | **SAM mask embeddings → InfoNCE contrastive on prototypes** |

Param count: roughly identical. Latency: identical (cosine sim ≈ dot product). The change is in *what the head learns*.

### Two-stage training recipe

**Stage 1 — Prototype pretraining via SAM (the novel part)**:

1. Run SAM-2/FastSAM on ~200K images → mask-per-object.
2. Sample positive embeddings: feature vectors at the **interior** of each mask (avoiding edges).
3. Sample negative embeddings: feature vectors at background pixels (>50px from any mask).
4. Train projection + prototypes via **InfoNCE loss**: positives close to *some* prototype, negatives far from *all* prototypes.
5. Freeze backbone during this stage. ~20 epochs, no bbox supervision yet.

**Stage 2 — Joint detection finetuning**:

1. Unfreeze backbone. Add bbox head.
2. Objectness loss: BCE on temperature-scaled prototype similarity (no Focal needed — similarity is naturally calibrated).
3. bbox loss: standard GIoU.
4. Mosaic + TAL soft labels + EMA. Standard recipe.

### Risk register (honest)

| Risk | Severity | Mitigation |
| --- | --- | --- |
| K=1 prototype too rigid for object diversity | High | Start K=4-8, ablate. Per-FPN-level prototypes if needed. |
| Prototypes collapse to backbone average | High | Orthogonality + diversity regularizers. Standard. |
| Cosine sim quantizes poorly to INT8 | Medium | Replace with `dot(x,p)/sqrt(D)` + learned scale — INT8-friendly. |
| Spatial locality lost (every position uses same global prototypes) | Medium | Per-FPN-level prototype sets (3 sets × 4 prototypes = 12 total params). |
| InfoNCE training unstable at tiny scale | Medium | SimCLR-style temperature + warmup, gradient clip. |
| Cross-domain transfer hypothesis fails empirically | **The bet** | This *is* the experiment. Null result still tells us something. |

### Honest framing of the publishability bar

**Plausibly paper-worthy** if it:

- Matches classifier-head baseline at equal params on COCO class-agnostic mAP (proves viability)
- AND transfers better cross-domain (e.g., +3+ mAP on road-signs after same finetune budget) (proves the inductive-bias claim)
- AND we ablate K, prototype regularizers, and pretrain data scale (so we know what matters)

**If only viability holds** → cool engineering, not a paper. Use it anyway.

**If nothing holds** → we publish the negative result (genuinely useful for the field), revert to classifier head, and still have the SAM-pretrain pipeline as a strong baseline.

### Connection to broader ML research themes

- **Metric learning revival** (prototypical networks, few-shot learning)
- **Self-supervised representation learning** for downstream detection (DINO, MAE)
- **Energy-based models** — if you squint, learned prototypes define an implicit energy landscape over feature space
- **In-context detection** — natural extension: swap prototypes at inference with user-provided exemplars → "detect things that look like this" without retraining. This is the long-term vision unlock.

---

## Pragmatic Innovations (the floor below PrototypeDet)

If PrototypeDet doesn't work, these still ship a competitive detector. They also stack with PrototypeDet if it does work.

### Why class-agnostic gives us leverage no published model uses

Every YOLO and every NanoDet is multi-class. They need labels for "car", "person", etc. We deleted the class head — which means:

1. **Any "this is an object" signal supervises us.** We're not constrained to labeled datasets.
2. **A teacher doesn't need to share our class set.** Any segmenter that outputs object regions can teach us.
3. **Foreground/background is the only confusion we fight** — not 80 mutually-exclusive classes.

This unlocks training data that multi-class models structurally cannot use.

### Headline: SAM-auto-labeled pretraining

**The bet**: pretrain BoxVision on **bounding boxes derived from SAM/SAM-2 segmentation masks** across 100K–1M images. SAM is class-agnostic by construction — it outputs masks for "anything that looks like an object" without ever needing a class label. Mask → bbox is trivial (axis-aligned bounding box of mask pixels).

**Why this beats COCO-collapsed-to-one-class as pretrain:**

- COCO has ~118K images and ~860K annotated objects, all from 80 specific categories.
- SAM-1B (Meta, 2023, open-licensed) has 11M images and **1.1 billion masks** across arbitrary objects — toys, parts, signage, screws, abstract shapes — far more diverse than COCO's category set.
- A 162K-param model is data-bound. Better supervision quality scales accuracy faster than architecture tweaks at this scale.

**Pipeline**:

1. Sample ~200K images from SA-1B (no labels needed, use the provided masks directly) plus optional crawl from CC-licensed sources.
2. Convert each SAM mask → bbox; filter by min area (e.g., 32×32 pixels) and IoU-dedup (drop boxes with >0.9 overlap to keep largest).
3. **Pretrain BoxVision-small for 30 epochs** on these auto-labels with mosaic + soft-label TAL recipe.
4. **Finetune for 50 epochs** on the actual target dataset (road-signs, or any downstream task).

**Expected gain**: +5-10 mAP vs ImageNet-backbone-only baseline on any downstream class-agnostic task. Bigger gain on domains visually distant from COCO.

**Cost**: One-time offline data prep (~1 day with FastSAM on a GPU). Pretraining takes ~2 days on a single GPU. Pretrained checkpoint is reusable across all downstream tasks.

**Risk**: SAM-derived boxes are noisy — SAM over-segments (sometimes one object becomes many parts). Mitigation: cluster nearby boxes during prep, use SAM-2's "everything" mode at moderate density, treat as weak supervision (soft labels in loss, not hard).

### Supporting innovation 1: Quantization-native architecture

**The bet**: design every layer to quantize cleanly to INT8 *from day one*, not retrofit after FP32 training.

**Specific changes**:

- Replace `nn.SiLU` (used in [fpn.py:30](boxvision/fpn.py#L30)) with `nn.ReLU` throughout neck + head. SiLU is non-monotonic and requires lookup tables in INT8.
- Keep all channel counts as multiples of 4 (already true: 32, 48, 96, 192) — aligns with AVX-VNNI 4-channel dot-product instruction.
- Replace `torch.exp(bbox_reg)` (the unfixed gap in [head.py:119](boxvision/head.py#L119)) with `F.softplus` or a learned scale — `exp` is fragile in INT8 and unbounded.
- Train with **QAT for last 20 epochs** of finetune. PyTorch `torch.ao.quantization` native flow.

**Goal**: INT8 mAP within 0.5 of FP32 mAP. Most detectors lose 2-5 mAP going to INT8 — being the model that doesn't is a real moat.

**Cost**: ~150 lines of QAT loop + activation observers, plus the ReLU swap.

### Supporting innovation 2: Edge-aware sub-pixel box refinement

**The bet**: class-agnostic accuracy is bottlenecked by box edge precision, not by classification. A tiny refinement head that snaps box edges to **image gradients** (raw RGB, not feature map) gives precision back at fractional cost.

**Specific mechanism**:

1. FCOS head outputs coarse box (x1, y1, x2, y2) at stride 8/16/32.
2. For each predicted edge, extract a thin strip of pixels from the **input image** perpendicular to the edge (e.g., 32 pixels wide × 4 pixels deep).
3. Tiny 1D conv stack (3 layers, 16 channels) predicts an offset in [−stride/2, +stride/2] per edge.
4. Apply offsets → refined box. Final NMS on refined boxes.

**Why this is novel for tiny detectors**: Cascade R-CNN and Mask R-CNN do similar refinement but at huge param cost. No <1M-param detector does edge-aware refinement on raw input pixels. For class-agnostic, where "what's the class" doesn't matter but "where exactly is the edge" matters a lot, this is high-leverage.

**Expected gain**: +2-4 mAP@0.7+ (the strict IoU thresholds where most tiny models fail), neutral on mAP@0.5.

**Cost**: ~5K params, ~1ms at 320×320 (one extra small op per detected box, gated by confidence threshold so it runs on <50 boxes typically).

### What we are NOT doing (and why)

- **NMS-free (one-to-one matching, YOLO10/26 style)**: defer. NMS overhead is <1ms at our scale; complexity > benefit for v3.
- **DETR-style query learning**: too much capacity needed; doesn't fit in 200K params.
- **Single-scale dilated head (Idea E above)**: tempting but loses small-object recall, and road-signs has small objects.
- **Density-prior auxiliary head**: redundant with SAM-pretrain — the SAM masks already encode density.

---

## v3 Roadmap — NanoDet-Plus Innovations to Adopt

NanoDet-Plus achieves +7 mAP over baseline NanoDet with these specific innovations. Reading their [code](https://github.com/RangiLyu/nanodet) is the source of truth — the README only describes them at a high level.

### 1. DSLA — Dynamic Soft Label Assigner (replaces TAL)

**Why**: For ultra-tiny models, hard top-k assignment (TAL) wastes signal. DSLA computes a continuous matching cost per (anchor, GT) pair and uses it as a soft weight in the loss, not a binary positive/negative.

**Expected gain**: +1-3 mAP for tiny models per NanoDet ablations.

**Cost**: ~80 lines in [losses.py](boxvision/losses.py), no inference impact.

### 2. AGM — Assign Guidance Module (training-only)

**Why**: Auxiliary head trained alongside the main model. The assigner uses AGM's predictions (not the main model's) to compute matching cost. AGM is discarded at export.

**Insight**: This is essentially **self-distillation embedded in the assignment step** — the model is supervised by a stronger version of itself. Free inference-time accuracy.

**Expected gain**: +2-3 mAP per NanoDet results.

**Cost**: ~120 lines (auxiliary head + assigner uses its output). Zero inference cost.

### 3. Ghost-PAN (consider — has tradeoffs)

**Why**: PAN = top-down + bottom-up, better small-object accuracy than top-down-only FPN. Ghost convs (primary + cheap "ghost" features) keep it light.

**Tradeoff**: We deliberately dropped bottom-up to save compute. Ghost-PAN costs ~5K extra params and ~1-2ms at 320. Worth it only if small-object recall is the bottleneck.

**Decision**: **Defer**. Validate top-down-only FPN first on road-signs. Adopt only if eval shows small-object recall is the limiter.

### 4. Generalized Focal Loss + DFL — DO NOT adopt

**Tension**: NanoDet uses GFL with distribution-based bbox regression. **YOLO26 explicitly removed DFL** for hardware compatibility / ONNX export simplicity.

**Decision**: **Stay aligned with YOLO26**. Plain GIoU + Focal. Distribution-based regression is hard to quantize cleanly and adds export complexity. We're optimizing for CPU INT8, not academic mAP.

---

## v3 Roadmap — Novel Innovations Worth Trying

Beyond NanoDet:

### 5. INT8 QAT with ShuffleNetV2 (highest-priority)

**Status**: Flag exists, implementation doesn't.

**Why**: 2-3× CPU speedup on x86 with VNNI, 1.5-2× on ARM NEON. Biggest single lever left.

**Approach**: PyTorch native quantization API (`torch.ao.quantization`), QAT for last 20 epochs. ShuffleNetV2 was chosen specifically because it has no SE blocks — should quantize cleanly.

**Risk**: Channel-shuffle op may not quantize cleanly in onnxruntime. Verify early with a smoke test before committing to QAT loop.

### 6. Knowledge Distillation from YOLO26m or YOLOv8m

**Why**: For models this small (162K params), distillation from a strong teacher consistently gives +2-4 mAP. Standard technique (FitNet, used in SimpleDet).

**Approach**: Teacher (frozen) produces dense objectness + box predictions on training images. Student matches teacher's outputs in addition to ground-truth loss.

**Cost**: ~100 lines added to [losses.py](boxvision/losses.py) + teacher forward pass during training (slows training, no inference impact).

### 7. RepConv reparameterization (consider)

**Why**: Train with multi-branch (3×3 + 1×1 + identity), fold to single 3×3 at inference. 1.2-1.5× speedup, no accuracy loss.

**Where to apply**: Head conv blocks. The FPN smoothing convs are already minimal — limited gain there.

**Cost**: ~60 lines for `RepConvBlock` + fold-on-export logic.

### 8. NMS-free via one-to-one matching — defer

**Why YOLO10/YOLO26 do this**: Removes NMS entirely, simplifies deployment.

**Why we shouldn't yet**: Requires careful training (two-head architecture during training, only one-to-one at inference). At our model scale, NMS overhead is small (<1ms on 100 candidates). Complexity > benefit for v3.

**Defer to v4** if NMS becomes a deployment blocker.

---

## Inference Runtime Strategy

### Runtime options researched

| Runtime | License | CPU strength | Verdict for us |
| --- | --- | --- | --- |
| **onnxruntime** | MIT | Solid, default | **Primary target.** Cross-platform, easy. |
| **OpenVINO** | Apache 2.0 | Best on Intel CPUs (VNNI, model compiler) | **Secondary target** — measure 2-3× speedup on x86 |
| **NCNN** | BSD | Excellent on ARM (mobile/Pi) | Optional — if we target ARM later |
| **MNN** | Apache 2.0 | Similar to NCNN, mobile-focused | Skip — NCNN covers same ground |
| **ailia SDK** | **Commercial / paid** | Vendor-claimed faster | **Skip** — not free, lock-in risk |

**Decision**: Export to ONNX → run on onnxruntime as primary, benchmark OpenVINO on Intel CPU for max-speed deployment. Skip ailia (it's a paid SDK from ax Inc, not a free runtime).

### ONNX export prerequisites

Before export works, refactor needed (see v2 Gap #3): move decode + NMS out of `forward()`. Export pure logits, do post-process in inference wrapper.

---

## Evaluation Strategy

### Metrics

- **Primary**: COCO mAP@0.5 and mAP@0.5:0.95 (class-agnostic)
- **Secondary**: Precision/Recall curves at threshold 0.35 (default deploy threshold)
- **Deployment**: CPU latency @ 320 and 640, FP32 + INT8, single-thread + 4-thread

### Implementation

- Use **pycocotools** for mAP — it's the standard, already in deps.
- Skip [rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics): PASCAL VOC only, no class-agnostic helper, no COCO 0.5:0.95.
- Write thin wrapper: collapse all predictions to category_id=1, evaluate against COCO-formatted GT with single category.

### Test Dataset

**Primary**: [Roboflow road-signs-6ih4y](https://universe.roboflow.com/roboflow-100/road-signs-6ih4y) — small, fast iteration, class-agnostic by collapsing classes.

**Stretch**: COCO val2017 with all classes collapsed — apples-to-apples vs YOLO26n.

---

## v3 Order of Operations

Sorted by ROI (impact ÷ effort). The headline innovation (SAM-pretrain) is sequenced **after** the cheap blockers because we need working training + export before any pretraining run is meaningful.

**Phase 0 — unblock (do first, ~1 day)**:

1. **Implement mosaic** in [dataset.py](boxvision/dataset.py) — claimed in config for 2 rounds, still missing. ~50 lines. Expected +2-4 mAP.
2. **Fix `exp` in [head.py:119](boxvision/head.py#L119)** — replace with `softplus` or `exp(clamp(x, max=8))`. Same fix gets us part of quantization-native goal.
3. **Refactor decode/NMS** out of `forward()` — required for ONNX export. ~30 lines.
4. **Train + eval baseline** on road-signs. **Get a real mAP number — everything below is speculation until then.**

**Phase 1 — headline innovation (the bet)**:

1. **SAM auto-label pipeline** — script to read SA-1B masks (or run FastSAM on unlabeled images) → bbox conversion → filtered training set. ~150 lines + offline run.
2. **SAM-pretrain BoxVision-small** for 30 epochs on auto-labeled data. Save as reusable checkpoint.
3. **Finetune from SAM-pretrain checkpoint** on road-signs. Compare delta vs ImageNet-only baseline.

**Phase 2 — supporting innovations**:

1. **Quantization-native swap** — ReLU instead of SiLU in neck/head, validate FP32 mAP unchanged.
2. **Edge-aware refinement head** — implement, train, ablate gain at IoU@0.7.
3. **INT8 QAT** — last 20 epochs of finetune with PyTorch quantization. Gate on INT8 mAP ≥ FP32 mAP − 0.5.

**Phase 3 — defensive upgrades**:

1. **AGM (auxiliary head)** from NanoDet — if SAM-pretrain alone doesn't hit target mAP.
2. **Knowledge distillation** from YOLO26m — if AGM doesn't close the gap.
3. **OpenVINO benchmark** — measure x86 speedup vs onnxruntime.
4. **RepConv** in head — 1.2-1.5× inference speedup, free accuracy.

Stop at any phase if mAP target is hit. Phases 2-3 are layered upgrades, not all required.

---

## Risks & Open Questions

- **162K params may be too small.** Head is 4.2K — that's ~26 params per output channel. Accuracy floor is real. Mitigation: try 2-conv head (still <10K), keep backbone same.
- **ShuffleNetV2 channel shuffle in ONNX.** Need to verify the channel-shuffle pattern exports as a clean `Reshape→Transpose→Reshape` that ORT can fuse. If not, switch to RepGhostNet (similar size, no shuffle).
- **DSLA + AGM are complex.** NanoDet-Plus's implementation is the reference. Easy to get wrong. Start with adopting DSLA alone, add AGM if first ablation lands.
- **Class-agnostic generalization on road-signs.** Road signs are visually distinct from COCO's "object" distribution. Pretrained ShuffleNetV2 (ImageNet) may not transfer well — could need higher LR on backbone or longer warmup.
- **mAP@0.5 vs mAP@0.5:0.95.** For class-agnostic, the @0.5 metric is more meaningful (don't need pixel-perfect boxes). Report both, optimize for @0.5.
- **YOLO26n apples-to-apples comparison.** YOLO26n is multi-class. Honest comparison requires either training YOLO26n class-agnostic ourselves, or accepting we measure on different objectives.
