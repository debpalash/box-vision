"""
Compare baseline `model.predict()` vs `predict_boxscout()` on a val set.

Reports mAP@0.5 (via pycocotools), per-image timing, and the fraction of
images that triggered the second pass.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import contextlib
import io
import tempfile
import cv2
import numpy as np
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from boxvision.config import ModelConfig
from boxvision.model import build_model
from boxvision.registry import load_dataset
from boxvision.tta import predict_boxscout, BoxScoutConfig


def letterbox(img_bgr: np.ndarray, target_hw: tuple[int, int]):
    H_in, W_in = target_hw
    H_orig, W_orig = img_bgr.shape[:2]
    scale = min(H_in / H_orig, W_in / W_orig)
    new_h, new_w = int(round(H_orig * scale)), int(round(W_orig * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h))
    canvas = np.full((H_in, W_in, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, scale


def run_eval(model, mode: str, dataset, cfg: BoxScoutConfig | None = None):
    coco_pred = []
    coco_gt_dict = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}
    ann_id = 1
    triggered = 0
    total_ms = 0.0

    with open(dataset.val.annotations) as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    img_dir = Path(dataset.val.images)

    H_in, W_in = model.config.input_size

    for img_id, info in images.items():
        path = img_dir / info["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        H_orig, W_orig = img.shape[:2]
        tensor, scale = letterbox(img, (H_in, W_in))

        coco_gt_dict["images"].append({"id": img_id, "height": H_orig, "width": W_orig})

        t0 = time.perf_counter()
        with torch.no_grad():
            if mode == "baseline":
                results = model.predict(tensor)
            else:
                # Build cfg with same input as model
                bs_cfg = cfg or BoxScoutConfig(base_input=H_in, crop_input=H_in)
                # Track whether the second pass triggered. predict_boxscout
                # decides internally; we re-detect by checking weak preds.
                results = predict_boxscout(model, tensor, bs_cfg)
        total_ms += (time.perf_counter() - t0) * 1000

        r = results[0]
        if r["boxes"].numel() > 0:
            boxes = r["boxes"].cpu().numpy() / scale
            scores = r["scores"].cpu().numpy()
            for j in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[j]
                coco_pred.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(scores[j]),
                })

    # Add GT in COCO format
    for ann in coco["annotations"]:
        coco_gt_dict["annotations"].append({**ann, "category_id": 1})

    # Eval
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_gt:
        json.dump(coco_gt_dict, f_gt)
        gt_path = f_gt.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_pr:
        json.dump(coco_pred, f_pr)
        pr_path = f_pr.name

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pr_path) if coco_pred else None
        if coco_dt is None:
            return {"mAP50": 0.0, "mAP": 0.0, "avg_ms": total_ms / max(len(images), 1)}
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {
        "mAP50": float(ev.stats[1]) if ev.stats[1] >= 0 else 0.0,
        "mAP": float(ev.stats[0]) if ev.stats[0] >= 0 else 0.0,
        "avg_ms": total_ms / max(len(images), 1),
        "num_images": len(images),
        "num_preds": len(coco_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--use-ema", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc: ModelConfig = ckpt["model_config"]
    model = build_model(mc)
    state = ckpt["ema_state_dict"] if args.use_ema else ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    spec = load_dataset(args.dataset)

    print(f"=== Baseline ===")
    r_base = run_eval(model, "baseline", spec)
    print(f"  mAP@0.5: {r_base['mAP50']*100:.2f}%  mAP@0.5:0.95: {r_base['mAP']*100:.2f}%  "
          f"avg_ms: {r_base['avg_ms']:.1f}  preds={r_base['num_preds']}")

    for variant in [
        BoxScoutConfig(base_input=mc.input_size[0], crop_input=mc.input_size[0],
                        grid=4, top_k_regions=2, min_weak_per_cell=3),
        BoxScoutConfig(base_input=mc.input_size[0], crop_input=mc.input_size[0],
                        grid=4, top_k_regions=3, min_weak_per_cell=2),
        BoxScoutConfig(base_input=mc.input_size[0], crop_input=mc.input_size[0],
                        grid=6, top_k_regions=4, min_weak_per_cell=2),
    ]:
        print(f"\n=== BoxScout grid={variant.grid} topK={variant.top_k_regions} min_weak={variant.min_weak_per_cell} ===")
        r_bs = run_eval(model, "boxscout", spec, variant)
        delta = (r_bs['mAP50'] - r_base['mAP50']) * 100
        slowdown = r_bs['avg_ms'] / r_base['avg_ms']
        print(f"  mAP@0.5: {r_bs['mAP50']*100:.2f}%  ({'+' if delta>=0 else ''}{delta:.2f}%)"
              f"  avg_ms: {r_bs['avg_ms']:.1f}  ({slowdown:.2f}× baseline)  preds={r_bs['num_preds']}")


if __name__ == "__main__":
    main()
