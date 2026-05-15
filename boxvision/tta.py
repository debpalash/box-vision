"""
Inference-time accuracy boosters for BoxVision.

Two innovations live here:

1. **predict_tta** — Test-time augmentation. Runs the model on the original
   image plus a horizontal-flipped copy and (optionally) multiple scales,
   merges all predictions with class-aware NMS. Doubles or quadruples
   inference cost; typically buys 1-3 mAP@0.5.

2. **predict_boxscout** — *Adaptive zoom cascade* (novel for class-agnostic
   detection). Always runs one pass at the model's native resolution to get
   coarse detections + an "uncertainty map" computed from weak predictions
   (objectness in [low_conf, high_conf]). Finds the top-K most uncertain
   regions and re-runs the model on those crops at the same input resolution
   — which effectively zooms the model 2-4× on regions where it most needs
   another look. Conditional compute: empty/confident images stay fast.

Both are pure inference helpers — no retraining required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torchvision.ops as ops

from .model import BoxVision


# ---------------------------------------------------------------------------
# TTA — horizontal flip + optional multi-scale
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_tta(
    model: BoxVision,
    image: torch.Tensor,
    scales: Optional[List[float]] = None,
    flip: bool = True,
    score_thresh: Optional[float] = None,
    nms_thresh: Optional[float] = None,
    max_dets: Optional[int] = None,
) -> List[dict]:
    """
    Test-time augmentation wrapper.

    Args:
        model:    BoxVision instance (will be put in eval mode)
        image:    [B, 3, H, W] float tensor, pre-normalized like training
        scales:   None ⇒ only run at native resolution; else a list of float
                  multipliers applied to (H, W). e.g. [1.0, 1.5] runs both.
        flip:     If True, also run a horizontally-flipped copy.
        score_thresh / nms_thresh / max_dets: Override model.config defaults.

    Returns:
        list[dict], one per batch element, with keys boxes / scores / labels.
    """
    model.eval()
    if scales is None:
        scales = [1.0]
    score_thresh = score_thresh if score_thresh is not None else model.config.objectness_threshold
    nms_thresh = nms_thresh if nms_thresh is not None else model.config.nms_threshold
    max_dets = max_dets if max_dets is not None else model.config.max_detections

    B, C, H, W = image.shape
    device = image.device
    per_batch_boxes: List[List[torch.Tensor]] = [[] for _ in range(B)]
    per_batch_scores: List[List[torch.Tensor]] = [[] for _ in range(B)]
    per_batch_labels: List[List[torch.Tensor]] = [[] for _ in range(B)]

    def _accumulate(results, flip_w: int = 0):
        for b, r in enumerate(results):
            boxes = r["boxes"]
            if boxes.numel() == 0:
                continue
            if flip_w:
                # Mirror x-coords back: x1' = W - x2, x2' = W - x1
                new = boxes.clone()
                new[:, 0] = flip_w - boxes[:, 2]
                new[:, 2] = flip_w - boxes[:, 0]
                boxes = new
            per_batch_boxes[b].append(boxes)
            per_batch_scores[b].append(r["scores"])
            per_batch_labels[b].append(r["labels"])

    for s in scales:
        if s == 1.0:
            img_s = image
        else:
            new_h = max(1, int(round(H * s)))
            new_w = max(1, int(round(W * s)))
            img_s = torch.nn.functional.interpolate(image, size=(new_h, new_w),
                                                    mode="bilinear", align_corners=False)
        out = model.predict(img_s)
        # Rescale boxes back to original image coords if we ran at a different size
        if s != 1.0:
            inv = 1.0 / s
            for r in out:
                r["boxes"] = r["boxes"] * inv
        _accumulate(out)

        if flip:
            img_flip = torch.flip(img_s, dims=[3])
            out_f = model.predict(img_flip)
            if s != 1.0:
                inv = 1.0 / s
                for r in out_f:
                    r["boxes"] = r["boxes"] * inv
            # Use flipped image's width to mirror coordinates back
            _accumulate(out_f, flip_w=img_s.shape[3] if s == 1.0 else W)

    # Merge per-batch via class-aware NMS
    results = []
    for b in range(B):
        if not per_batch_boxes[b]:
            results.append({
                "boxes": torch.zeros((0, 4), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
            })
            continue
        boxes = torch.cat(per_batch_boxes[b], dim=0)
        scores = torch.cat(per_batch_scores[b], dim=0)
        labels = torch.cat(per_batch_labels[b], dim=0)
        keep = scores > score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if boxes.numel() > 0:
            keep_idx = ops.batched_nms(boxes, scores, labels, nms_thresh)
            keep_idx = keep_idx[:max_dets]
            boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
        results.append({"boxes": boxes, "scores": scores, "labels": labels})
    return results


# ---------------------------------------------------------------------------
# BoxScout — adaptive zoom cascade
# ---------------------------------------------------------------------------

@dataclass
class BoxScoutConfig:
    """Tuning knobs for the adaptive zoom cascade."""
    base_input: int = 416           # native resolution for pass 1
    low_conf: float = 0.10          # weak-prediction band (lower bound)
    high_conf: float = 0.40         # weak-prediction band (upper bound)
    grid: int = 4                    # NxN cell grid over the image for uncertainty map
    min_weak_per_cell: int = 3       # cells with this many weak preds become hot
    top_k_regions: int = 3           # max crops per image
    crop_pad: float = 0.15           # pad each hot region by this fraction
    crop_input: int = 416            # resize crops to this size for re-detection


@torch.no_grad()
def predict_boxscout(
    model: BoxVision,
    image: torch.Tensor,
    cfg: Optional[BoxScoutConfig] = None,
) -> List[dict]:
    """
    Adaptive two-pass detection. Runs the model at base resolution, then
    re-runs on the top-K most uncertain spatial regions at zoomed scale.

    Inputs:
        model: BoxVision (eval mode is forced)
        image: [B, 3, H, W] preprocessed (already letterboxed + normalized)
        cfg:   BoxScoutConfig (or default)

    Returns:
        list[dict] with merged boxes/scores/labels per batch element.
    """
    model.eval()
    cfg = cfg or BoxScoutConfig()
    B, _, H, W = image.shape
    device = image.device

    # ----- Pass 1: native-resolution detection + uncertainty map -----
    # We need access to ALL predictions, including weak ones, to build the
    # uncertainty map. Temporarily lower the score threshold for that purpose.
    saved_thresh = model.config.objectness_threshold
    model.config.objectness_threshold = cfg.low_conf
    pass1 = model.predict(image)
    model.config.objectness_threshold = saved_thresh

    results: List[dict] = []
    for b in range(B):
        r = pass1[b]
        boxes, scores, labels = r["boxes"], r["scores"], r["labels"]

        # Confident (≥ saved_thresh): keep as final output.
        strong_mask = scores >= saved_thresh
        strong_boxes = boxes[strong_mask]
        strong_scores = scores[strong_mask]
        strong_labels = labels[strong_mask]

        # Weak predictions (low ≤ score < high): drive the uncertainty map.
        weak_mask = (scores >= cfg.low_conf) & (scores < cfg.high_conf)
        weak_boxes = boxes[weak_mask]

        # Build NxN density grid over the weak predictions' centers
        regions = _hot_regions(weak_boxes, H, W, cfg)

        # ----- Pass 2: re-run on each hot region (zoomed) -----
        for (rx1, ry1, rx2, ry2) in regions:
            crop_h = ry2 - ry1
            crop_w = rx2 - rx1
            if crop_h < 8 or crop_w < 8:
                continue
            crop = image[b:b + 1, :, ry1:ry2, rx1:rx2]
            crop_resized = torch.nn.functional.interpolate(
                crop, size=(cfg.crop_input, cfg.crop_input),
                mode="bilinear", align_corners=False,
            )
            pass2 = model.predict(crop_resized)[0]
            if pass2["scores"].numel() == 0:
                continue
            # Map crop coordinates back to the original image.
            sx = crop_w / cfg.crop_input
            sy = crop_h / cfg.crop_input
            cb = pass2["boxes"].clone()
            cb[:, 0] = cb[:, 0] * sx + rx1
            cb[:, 1] = cb[:, 1] * sy + ry1
            cb[:, 2] = cb[:, 2] * sx + rx1
            cb[:, 3] = cb[:, 3] * sy + ry1
            strong_boxes = torch.cat([strong_boxes, cb], dim=0)
            strong_scores = torch.cat([strong_scores, pass2["scores"]], dim=0)
            strong_labels = torch.cat([strong_labels, pass2["labels"]], dim=0)

        # Final class-aware NMS to deduplicate overlap between pass-1 strongs
        # and pass-2 finds.
        if strong_boxes.numel() > 0:
            keep = ops.batched_nms(strong_boxes, strong_scores, strong_labels,
                                    model.config.nms_threshold)
            keep = keep[: model.config.max_detections]
            strong_boxes = strong_boxes[keep]
            strong_scores = strong_scores[keep]
            strong_labels = strong_labels[keep]

        results.append({
            "boxes": strong_boxes,
            "scores": strong_scores,
            "labels": strong_labels,
        })
    return results


def _hot_regions(weak_boxes: torch.Tensor, H: int, W: int,
                 cfg: BoxScoutConfig) -> List[Tuple[int, int, int, int]]:
    """Identify up to top_k_regions [x1, y1, x2, y2] crops that contain the most
    weak predictions. Returns image-coordinate integer tuples."""
    if weak_boxes.numel() == 0:
        return []
    # Use box centers
    cx = (weak_boxes[:, 0] + weak_boxes[:, 2]) * 0.5
    cy = (weak_boxes[:, 1] + weak_boxes[:, 3]) * 0.5
    gx = (cx / W * cfg.grid).clamp(0, cfg.grid - 1).long()
    gy = (cy / H * cfg.grid).clamp(0, cfg.grid - 1).long()
    counts = torch.zeros((cfg.grid, cfg.grid), dtype=torch.long, device=weak_boxes.device)
    for x, y in zip(gx.tolist(), gy.tolist()):
        counts[y, x] += 1

    flat = counts.flatten()
    # Pick top-K cells with at least min_weak_per_cell weak predictions
    hot = []
    sorted_idx = torch.argsort(flat, descending=True)
    cell_h = H / cfg.grid
    cell_w = W / cfg.grid
    pad_h = cell_h * cfg.crop_pad
    pad_w = cell_w * cfg.crop_pad
    for idx in sorted_idx[: cfg.top_k_regions].tolist():
        if flat[idx].item() < cfg.min_weak_per_cell:
            break
        cy_grid = idx // cfg.grid
        cx_grid = idx % cfg.grid
        x1 = int(max(0, cx_grid * cell_w - pad_w))
        y1 = int(max(0, cy_grid * cell_h - pad_h))
        x2 = int(min(W, (cx_grid + 1) * cell_w + pad_w))
        y2 = int(min(H, (cy_grid + 1) * cell_h + pad_h))
        hot.append((x1, y1, x2, y2))
    return hot
