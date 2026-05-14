# BoxVision — Ecosystem Blueprint

This is the system-level map of BoxVision. [goal.md](goal.md) is the model/research plan; this document is the surrounding ecosystem — data, training, deployment, tooling, integrations, releases.

**Read order**: this doc explains *what we are building beyond the model itself*. [goal.md](goal.md) covers the model architecture and research direction (PrototypeDet).

---

## What BoxVision is, who it's for

**BoxVision** is a class-agnostic bounding-box detector optimized for CPU inference. It finds objects in an image without telling you what they are.

**Target users**:

- Engineers building anomaly detection / "find all objects" pipelines where downstream logic handles classification.
- Robotics / drone / embedded developers on CPU-only platforms where YOLO is too heavy.
- ML platform teams using BoxVision as a fast pre-filter before a downstream classifier or VLM.
- Researchers who want a baseline class-agnostic detector to ablate against.

**Value proposition vs alternatives**:

| Use case | Today's option | BoxVision pitch |
| --- | --- | --- |
| Real-time CPU detection on i7/M1 | YOLO11n / YOLO26n (~30-40ms ONNX) | **~6-10ms ONNX at comparable recall** for object localization |
| Class-agnostic prefilter | YOLO with `nc=1` (still expensive) | **15× fewer params**, purpose-built for the task |
| Browser/WASM detection | No good option in 2026 | **<3MB INT8 model** fits comfortably |
| Cross-domain transfer | YOLO finetune (5-10K labels needed) | **SAM-pretrained checkpoint** transfers with fewer labels |
| Open-vocabulary detection (long-term) | Grounding DINO, OWL-ViT (huge) | **Prototype swap at inference** for in-context detection |

---

## System diagram

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA ECOSYSTEM                               │
│                                                                      │
│  ┌────────────┐   ┌──────────────┐   ┌─────────────────────────────┐ │
│  │ SA-1B      │   │ Roboflow     │   │ Custom data (any source)    │ │
│  │ (pretrain) │   │ (finetune)   │   │ + FastSAM auto-labeling     │ │
│  └─────┬──────┘   └──────┬───────┘   └──────────────┬──────────────┘ │
│        │                 │                          │                │
│        v                 v                          v                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                  Dataset conversion layer                      │  │
│  │  - boxvision.data.coco_loader (current)                        │  │
│  │  - boxvision.data.yolo_loader (planned)                        │  │
│  │  - boxvision.data.sam_autolabel (planned)                      │  │
│  │  - boxvision.data.roboflow_restructure (planned)               │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
└─────────────────────────────────│────────────────────────────────────┘
                                  │
                                  v
┌──────────────────────────────────────────────────────────────────────┐
│                       MODEL & TRAINING                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ BoxVision-tiny  (162K, 0.62 MB)   target: 8-10ms ONNX @ 320     │ │
│  │ BoxVision-small (838K, 3.2  MB)   target: 15-18ms ONNX @ 320    │ │
│  │ BoxVision-pico  (<50K)            future: <5ms, edge devices    │ │
│  │ BoxVision-base  (~3M)             future: highest accuracy tier │ │
│  └────────────────────────────────┬────────────────────────────────┘ │
│                                   │                                  │
│  ┌────────────────────────────────v───────────────────────────────┐  │
│  │ Training pipeline                                              │  │
│  │  - Stage 1: SAM-contrastive prototype pretrain                 │  │
│  │  - Stage 2: detection finetune (mosaic, soft TAL, EMA)         │  │
│  │  - Stage 3 (optional): INT8 QAT for last 20 epochs             │  │
│  │  - Stage 4 (optional): teacher distillation from YOLO26m       │  │
│  └────────────────────────────────┬───────────────────────────────┘  │
└───────────────────────────────────│──────────────────────────────────┘
                                    │
                                    v
┌──────────────────────────────────────────────────────────────────────┐
│                          EXPORT & DEPLOYMENT                         │
│                                                                      │
│   PyTorch checkpoint (.pt)                                           │
│        │                                                             │
│        ├──> ONNX FP32  ──> onnxruntime (primary, all platforms)      │
│        ├──> ONNX INT8  ──> onnxruntime + OpenVINO (Intel CPU)        │
│        ├──> NCNN model ──> mobile / ARM (Pi, Android)                │
│        └──> Web bundle ──> onnxruntime-web (browser, WASM+SIMD)      │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌──────────────────────────────────────────────────────────────────────┐
│                       USER-FACING SURFACES                           │
│                                                                      │
│   - Python package   (pip install boxvision)                         │
│   - CLI              (boxvision train/export/detect/benchmark)       │
│   - Docker image     (training-ready, all deps)                      │
│   - HuggingFace Hub  (model weights + cards)                         │
│   - Examples repo    (road-signs, COCO, custom-data tutorials)       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Module breakdown

Status legend: ✅ built · 🚧 in-progress · 📋 planned · ❓ uncertain

### Core model (boxvision package)

| Module | File | Status | Notes |
| --- | --- | --- | --- |
| Backbone | [backbone.py](boxvision/backbone.py) | ✅ | ShuffleNetV2 0.5x / 1.0x, dual variants |
| Neck (FPN/Ghost-FPN) | [fpn.py](boxvision/fpn.py) | ✅ | Top-down only; Ghost modules for `small` preset |
| Head | [head.py](boxvision/head.py) | ✅ | 1-conv FCOS, class-agnostic. **`exp` clamp needed.** |
| Head (Prototype) | [head.py](boxvision/head.py) | 📋 | PrototypeDet replacement — see [goal.md](goal.md) |
| Losses | [losses.py](boxvision/losses.py) | ✅ | TAL with soft IoU labels (DSLA-flavored) |
| Losses (InfoNCE) | [losses.py](boxvision/losses.py) | 📋 | For prototype pretraining |
| Model assembly | [model.py](boxvision/model.py) | ✅ | + EMA. **Decode/NMS still in `forward()` — needs refactor for ONNX.** |
| Config / presets | [config.py](boxvision/config.py) | ✅ | tiny/small factories |

### Data ecosystem

| Module | Status | Notes |
| --- | --- | --- |
| Dataset registry | ✅ | [datasets.yaml](datasets.yaml) + [registry.py](boxvision/registry.py). `boxvision datasets` to list. |
| COCO loader | ✅ in [dataset.py](boxvision/dataset.py) | Path-agnostic (registry resolves). Roboflow + standard COCO + custom all work. |
| YOLO TXT loader | 📋 | Entry registered (`road-signs-yolo`); loader implementation pending. Will raise NotImplementedError until added. |
| Roboflow restructure script | ❌ not needed | Replaced by registry — Roboflow's native layout is registered directly. |
| Mosaic augmentation | ✅ | Real impl in [dataset.py:111](boxvision/dataset.py#L111), 80% probability, `set_mosaic()` for late-epoch toggle. |
| SAM auto-label pipeline | 📋 | `scripts/sam_autolabel.py` — read SA-1B masks or run FastSAM → bbox-per-mask. Output registered as new dataset entry. |
| Class-agnostic collapse | ✅ implicit | All loaders set `label=0` regardless of category. Controlled by `class_agnostic: true` in registry entries. |

### Training pipeline

| Component | Status | Notes |
| --- | --- | --- |
| Standard finetune loop | ✅ in [train.py](boxvision/train.py) | SGD + cosine, EMA, AMP |
| SAM-contrastive pretrain loop | 📋 | Stage 1 of PrototypeDet; separate entry point |
| Distillation hooks | 📋 | Teacher forward in training step; output matching loss |
| INT8 QAT loop | 📋 | PyTorch `torch.ao.quantization` integration |
| Experiment tracking | 📋 | Decide: W&B (cloud) vs MLflow (self-host) vs plain CSV |
| Seed control / determinism | 📋 | `torch.use_deterministic_algorithms(True)` + seed-everything helper |
| Checkpoint format | ✅ | `.pt` with model_state + ema_state + config dataclasses |

### Export & runtime

| Target | Status | Notes |
| --- | --- | --- |
| ONNX export | 🚧 in [export.py](boxvision/export.py) | Will fail until decode/NMS moved out of `forward()` |
| onnxruntime inference | ✅ in [export.py](boxvision/export.py) | Primary path |
| OpenVINO export | 📋 | Convert ONNX → OpenVINO IR; benchmark on Intel CPU |
| NCNN export | 📋 | ONNX → NCNN via `onnx2ncnn`; ARM benchmark |
| onnxruntime-web | 📋 | WASM+SIMD; bundle for browser demo |
| INT8 quantization (post-training) | 🚧 flag only | `--quantize` flag in CLI, calibration loop missing |
| INT8 (QAT) | 📋 | Higher-fidelity than PTQ; deferred to Phase 2 |

### Tooling / CLI

| Command | Status | Notes |
| --- | --- | --- |
| `boxvision train --dataset <name>` | ✅ | Resolves via registry |
| `boxvision datasets` | ✅ | List registered datasets + ready/missing status |
| `boxvision export` | ✅ | Will work once ONNX refactor lands |
| `boxvision detect` | ✅ | Single-image inference |
| `boxvision benchmark` | ✅ | Latency stats |
| `boxvision info` | ✅ | Print model architecture |
| `boxvision autolabel` | 📋 | Run SAM + write predictions in COCO format |
| `boxvision visualize` | 📋 | Render predictions + (for PrototypeDet) prototype attention maps |
| `boxvision profile` | 📋 | Per-op latency breakdown (helps find bottlenecks) |

### Integrations

| Integration | Status | Notes |
| --- | --- | --- |
| Python package | ✅ via `pyproject.toml` | `uv pip install -e .` works |
| PyPI release | 📋 | Defer until first stable checkpoint exists |
| Docker training image | 📋 | CUDA + uv + pinned deps |
| HuggingFace Hub | 📋 | Model card + weights per variant per pretrain |
| Roboflow Universe upload | ❓ | Could publish as a community model |
| Ultralytics interop shim | ❓ | "Use BoxVision in YOLO pipelines" — niche, low priority |

### Documentation

| Doc | Status | Notes |
| --- | --- | --- |
| README.md | ✅ basic | Needs quickstart update post-Phase 0 |
| goal.md | ✅ | Architecture + research plan (this file's sibling) |
| BLUEPRINT.md | ✅ | This document |
| Training guide | 📋 | End-to-end walkthrough |
| Deployment guide (per runtime) | 📋 | One per: onnxruntime, OpenVINO, NCNN, web |
| API reference | 📋 | Sphinx or mkdocs |
| Examples (`examples/`) | 📋 | road-signs, COCO-collapsed, custom-data, prototype-swap demo |

---

## Use cases the ecosystem targets

### Tier 1 — confirmed during planning

1. **Road-signs detection (test bed)**. 21 classes collapsed to 1. Validates the class-agnostic pipeline end-to-end.
2. **Generic object detection pre-filter**. Run BoxVision → crop boxes → pass to a classifier or VLM. Saves compute on empty regions.
3. **Anomaly localization**. "Find everything that's an object, flag what shouldn't be there." Quality control, security, surveillance.

### Tier 2 — natural extensions

4. **Auto-labeling helper**. Use BoxVision as a coarse box-proposer in a human-in-the-loop labeling tool (Roboflow / Label Studio plugin).
5. **Embedded vision**. Raspberry Pi 4, Jetson Nano, low-power microcontrollers — INT8 + NCNN deployment.
6. **Browser-based demos**. <3MB INT8 model loads fast over network; runs in WebAssembly with SIMD.

### Tier 3 — speculative / research

7. **In-context detection** (long-term, PrototypeDet enables this). User provides 3-5 examples of what they want → swap into prototype slots → detect at inference without retraining.
8. **Multi-modal pre-stage**. BoxVision provides region proposals to a VLM (which is too slow to dense-predict). Boxes become "where to look" prompts.

---

## Release / versioning strategy

### Model variants and naming

```text
boxvision-{size}-{pretrain}-{epoch}.pt

  size      = tiny | small | base | pico
  pretrain  = imagenet | sam | sam-finetune | coco-agnostic
  epoch     = epoch number or "best"
```

Examples:

- `boxvision-tiny-imagenet-best.pt` — backbone pretrained on ImageNet only
- `boxvision-small-sam-30.pt` — Stage 1 SAM-pretrained, 30 epochs
- `boxvision-small-sam-finetune-roadsigns-best.pt` — Stage 2 finetune output

### Semver for the package

- **0.x** — current, breaking changes allowed
- **1.0** — first stable release; gating criteria:
  - All Phase 0 + Phase 1 (SAM pretrain) shipped
  - ONNX export validated
  - At least one published checkpoint per size
  - Examples directory with 3+ working tutorials

### License

- **Code**: MIT (recommended; user to confirm)
- **Trained weights**: needs care — SAM-1B is Apache 2.0 but Meta's research-use clause applies; document carefully

---

## Open questions / decisions to lock

| Question | Default leaning | Decide by |
| --- | --- | --- |
| Experiment tracking: W&B vs MLflow vs CSV? | **W&B** for cloud convenience, free tier | Before first multi-day training run |
| Distribute weights via HuggingFace or own bucket? | **HuggingFace** — discoverable, free | Before first public release |
| Ship `boxvision` on PyPI or only GitHub install? | **PyPI** once API stabilizes | After v0.5 |
| Should the package include pre-trained weights or fetch on demand? | **Fetch on demand** — smaller package | At PyPI release time |
| Documentation: Sphinx vs MkDocs vs plain markdown? | **MkDocs** (modern, simpler) | Before v1.0 |
| Citation / paper venue? | **arXiv preprint first**, conference TBD | After empirical validation of PrototypeDet |
| License of weights vs license of code | Probably different — needs lawyer-grade phrasing | Before any public weight release |

---

## Build order (dependency-aware)

This is the actual sequence of work, with dependencies. Compare against [goal.md](goal.md)'s research-prioritized order — this one is engineering-prioritized.

**M0 — Foundation (unblock everything)**

1. ✅ Mosaic + `exp` clamp from [goal.md](goal.md) Phase 0 blockers
2. ❌ ONNX refactor (decode/NMS out of `forward()`) — still pending
3. ✅ Dataset registry ([datasets.yaml](datasets.yaml) + [registry.py](boxvision/registry.py)) — replaces planned Roboflow prep script
4. 🚧 Train + eval baseline on road-signs → first real mAP number

**M1 — First public checkpoint**

1. SAM auto-label pipeline + Stage 1 pretrain (could use FastSAM if SA-1B is too heavy)
2. Stage 2 finetune on road-signs from SAM-pretrained checkpoint
3. Publish `boxvision-small-sam-roadsigns-v0.1` checkpoint somewhere (HF Hub or own bucket)

**M2 — Deployment surfaces**

1. ONNX FP32 export validated, latency measured
2. ONNX INT8 (PTQ) + calibration loop
3. OpenVINO conversion + Intel CPU benchmark
4. README example: load checkpoint → run inference → save annotated image

**M3 — Research credibility (the PrototypeDet bet)**

1. Implement Prototype head as opt-in (alongside classifier head for ablation)
2. InfoNCE pretrain loop
3. Ablation: classifier vs prototype on same data, same params
4. Cross-domain transfer experiment (train on COCO-collapsed, test on road-signs / custom domain)
5. Writeup — blog post or arXiv preprint depending on result strength

**M4 — Ecosystem (post-research)**

1. Browser/WASM bundle
2. NCNN export for ARM
3. PyPI release
4. Docker image
5. MkDocs site

---

## What this blueprint is NOT

- **Not a research plan** — that's [goal.md](goal.md). PrototypeDet rationale, risk register, and architectural detail live there.
- **Not a marketing doc** — claims are calibrated to what we've measured (162K params, 13ms PyTorch) or what's gated by experiments (SAM pretrain gains, prototype-head viability).
- **Not a frozen spec** — every "📋 planned" item is a decision waiting to be made. Cut anything that doesn't pull weight.
