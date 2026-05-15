"""
Failure analysis for a trained BoxVision checkpoint.

Iterates the validation set, runs predictions, classifies each GT and each
prediction as TP / FN / FP at IoU=0.5, and reports stats sliced by box size
and (for shapes) original class. Also dumps the N worst-case images with GT
in green and predictions in red overlaid.

Usage:
    python scripts/analyze_failures.py \
        --checkpoint runs/shapes-tiny-100ep/best.pt \
        --dataset shapes \
        --out runs/shapes-tiny-100ep/failures \
        --top-n 30
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image

from boxvision.config import ModelConfig
from boxvision.model import build_model
from boxvision.registry import load_dataset


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two box sets in (x1, y1, x2, y2) — [N,4] vs [M,4] -> [N,M]."""
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    a_e = a[:, None, :]
    b_e = b[None, :, :]
    x1 = np.maximum(a_e[..., 0], b_e[..., 0])
    y1 = np.maximum(a_e[..., 1], b_e[..., 1])
    x2 = np.minimum(a_e[..., 2], b_e[..., 2])
    y2 = np.minimum(a_e[..., 3], b_e[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-7)


def size_bucket(box: np.ndarray) -> str:
    w = box[2] - box[0]
    h = box[3] - box[1]
    area = w * h
    if area < 32 * 32:
        return "small (<32²)"
    if area < 96 * 96:
        return "medium (32²-96²)"
    return "large (≥96²)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.35,
                    help="Confidence threshold for inference (not for mAP eval)")
    ap.add_argument("--top-n", type=int, default=30, help="Save N worst images")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg: ModelConfig = ckpt["model_config"]
    model_cfg.objectness_threshold = args.conf
    model = build_model(model_cfg)
    state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    print(f"Checkpoint: epoch {ckpt['epoch']+1}, input {model_cfg.input_size}")

    # Load dataset
    spec = load_dataset(args.dataset)
    coco_path = Path(spec.val.annotations)
    with open(coco_path) as f:
        coco = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}
    # NOTE: shapes was trained class-agnostic but the JSON has a single 'object'
    # category. To analyze per-original-class, reread the labels in
    # datasets/ls-project615 if needed. Here we work with stored categories.
    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_by_image[ann["image_id"]].append({
            "box": np.array([x, y, x + w, y + h], dtype=np.float32),
            "cat": cat_by_id.get(ann["category_id"], "object"),
        })

    # Stats accumulators
    n_tp = 0
    n_fp = 0
    n_fn = 0
    by_size = defaultdict(lambda: {"tp": 0, "fn": 0})
    images_zero_pred = 0
    images_with_misses: list[tuple[float, int]] = []  # (miss_score, image_id)
    image_dir = Path(spec.val.images)

    H_in, W_in = model_cfg.input_size

    for img_id, img_info in images.items():
        fname = img_info["file_name"]
        img_path = image_dir / fname
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        H_orig, W_orig = img_bgr.shape[:2]

        # Match the same letterbox transform the loader uses (LongestMaxSize + Pad).
        # Simpler approximation: scale longest edge to H_in, pad with 114.
        scale = min(H_in / H_orig, W_in / W_orig)
        new_h = int(round(H_orig * scale))
        new_w = int(round(W_orig * scale))
        resized = cv2.resize(img_bgr, (new_w, new_h))
        canvas = np.full((H_in, W_in, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)

        with torch.no_grad():
            results = model.predict(tensor)
        result = results[0]
        pred_boxes = result["boxes"].cpu().numpy()  # in network coords
        pred_scores = result["scores"].cpu().numpy()

        # Map predictions back to original image coords (undo letterbox).
        if pred_boxes.shape[0] > 0:
            pred_boxes_orig = pred_boxes.copy()
            pred_boxes_orig[:, [0, 2]] /= scale
            pred_boxes_orig[:, [1, 3]] /= scale
            pred_boxes_orig[:, [0, 2]] = np.clip(pred_boxes_orig[:, [0, 2]], 0, W_orig)
            pred_boxes_orig[:, [1, 3]] = np.clip(pred_boxes_orig[:, [1, 3]], 0, H_orig)
        else:
            pred_boxes_orig = np.zeros((0, 4), dtype=np.float32)

        gts = gt_by_image.get(img_id, [])
        gt_boxes = np.array([g["box"] for g in gts], dtype=np.float32) if gts else np.zeros((0, 4), dtype=np.float32)

        if pred_boxes_orig.shape[0] == 0 and gt_boxes.shape[0] > 0:
            images_zero_pred += 1

        ious = iou_xyxy(pred_boxes_orig, gt_boxes)  # [P, G]
        matched_gt = np.zeros(gt_boxes.shape[0], dtype=bool)
        matched_pred = np.zeros(pred_boxes_orig.shape[0], dtype=bool)
        # Greedy match by descending pred score
        order = np.argsort(-pred_scores) if pred_scores.size > 0 else np.array([], dtype=int)
        for pi in order:
            if gt_boxes.shape[0] == 0:
                break
            best_g = int(np.argmax(ious[pi]))
            if ious[pi, best_g] >= args.iou and not matched_gt[best_g]:
                matched_gt[best_g] = True
                matched_pred[pi] = True

        img_tp = int(matched_pred.sum())
        img_fp = int((~matched_pred).sum())
        img_fn = int((~matched_gt).sum())
        n_tp += img_tp
        n_fp += img_fp
        n_fn += img_fn

        for g_idx, gt in enumerate(gts):
            bkt = size_bucket(gt["box"])
            if matched_gt[g_idx]:
                by_size[bkt]["tp"] += 1
            else:
                by_size[bkt]["fn"] += 1

        # Worst-case score = FN count + 0.1 * FP count (recall is the bigger story here)
        miss_score = img_fn + 0.1 * img_fp
        images_with_misses.append((miss_score, img_id, fname, gts, pred_boxes_orig, pred_scores, matched_gt, matched_pred))

    # Report
    precision = n_tp / max(n_tp + n_fp, 1)
    recall = n_tp / max(n_tp + n_fn, 1)
    print(f"\n=== Overall @ IoU≥{args.iou}, conf≥{args.conf} ===")
    print(f"  TP: {n_tp}   FP: {n_fp}   FN: {n_fn}")
    print(f"  Precision: {precision*100:.2f}%   Recall: {recall*100:.2f}%")
    print(f"  Images with zero predictions: {images_zero_pred}/{len(images)}")
    print(f"\n=== By box size ===")
    for bkt in ["small (<32²)", "medium (32²-96²)", "large (≥96²)"]:
        st = by_size[bkt]
        tot = st["tp"] + st["fn"]
        rec = st["tp"] / max(tot, 1)
        print(f"  {bkt:22s}  TP={st['tp']:4d}  FN={st['fn']:4d}  recall={rec*100:.1f}%  (n={tot})")

    # Save worst-case viz
    images_with_misses.sort(key=lambda r: -r[0])
    print(f"\nSaving top {args.top_n} worst cases to {out}/")
    for score, img_id, fname, gts, preds, pscores, mg, mp in images_with_misses[: args.top_n]:
        if score == 0:
            break
        img_path = image_dir / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for g_idx, gt in enumerate(gts):
            x1, y1, x2, y2 = map(int, gt["box"])
            color = (0, 200, 0) if mg[g_idx] else (0, 0, 255)  # green TP, red missed
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        for pi in range(preds.shape[0]):
            x1, y1, x2, y2 = map(int, preds[pi])
            color = (200, 200, 0) if mp[pi] else (255, 0, 200)  # yellow TP-pred, magenta FP
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(img, f"{pscores[pi]:.2f}", (x1, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(str(out / f"miss_{int(score*10):03d}_{fname}"), img)

    print("\nLegend on saved images:")
    print("  green   = TP (GT matched)")
    print("  red     = FN (GT missed)")
    print("  yellow  = TP prediction")
    print("  magenta = FP (spurious prediction)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
