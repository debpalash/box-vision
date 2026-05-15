"""
Generate teacher pseudo-labels for a training set, then write an augmented
COCO JSON that combines real GT with teacher-discovered boxes.

For class-agnostic detection on a small dataset, the teacher's extra boxes
act as ~30-50% more training signal. The student then trains on this
combined supervision and inherits the teacher's "small-object eye" even
though it's deployed at lower resolution.

Usage:
    python scripts/distill_pseudo_labels.py \
        --teacher runs/H-teacher-small-640-300ep/best.pt \
        --dataset shapes \
        --out datasets/shapes-distilled \
        --conf 0.40 \
        --iou-with-gt 0.5

The output folder mirrors the original dataset (train/_annotations.coco.json
+ images + valid/...), with the train JSON augmented. Valid stays untouched.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from boxvision.model import build_model
from boxvision.registry import load_dataset


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_e, b_e = a[:, None, :], b[None, :, :]
    x1 = np.maximum(a_e[..., 0], b_e[..., 0])
    y1 = np.maximum(a_e[..., 1], b_e[..., 1])
    x2 = np.minimum(a_e[..., 2], b_e[..., 2])
    y2 = np.minimum(a_e[..., 3], b_e[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(a_area[:, None] + b_area[None, :] - inter, 1e-7)


def letterbox(img, size):
    H_in, W_in = size, size
    H_o, W_o = img.shape[:2]
    s = min(H_in / H_o, W_in / W_o)
    nh, nw = int(round(H_o * s)), int(round(W_o * s))
    r = cv2.resize(img, (nw, nh))
    canvas = np.full((H_in, W_in, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = r
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, help="Output dataset root dir")
    ap.add_argument("--conf", type=float, default=0.40, help="Min teacher score to keep")
    ap.add_argument("--iou-with-gt", type=float, default=0.5,
                    help="If teacher box overlaps a real GT by > this, drop it (real wins)")
    ap.add_argument("--max-per-image", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    mc = ckpt["model_config"]
    mc.objectness_threshold = args.conf
    model = build_model(mc).to(device).eval()
    model.load_state_dict(ckpt.get("ema_state_dict") or ckpt["model_state_dict"])
    teacher_size = mc.input_size[0]
    print(f"Teacher: epoch {ckpt['epoch']+1}, input {mc.input_size}, conf={args.conf}")

    spec = load_dataset(args.dataset)
    train_dir = Path(spec.train.images)
    train_ann = Path(spec.train.annotations)
    valid_dir = Path(spec.val.images)
    valid_ann = Path(spec.val.annotations)

    out = Path(args.out)
    out_train = out / "train"
    out_valid = out / "valid"
    out_train.mkdir(parents=True, exist_ok=True)
    out_valid.mkdir(parents=True, exist_ok=True)

    # Copy images and annotations files for valid (unchanged)
    for p in valid_dir.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(p, out_valid / p.name)
    shutil.copy2(valid_ann, out_valid / "_annotations.coco.json")

    # Process train: copy images, augment annotations
    with open(train_ann) as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    gt_by_img = {}
    for ann in coco["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)
    max_ann_id = max((ann["id"] for ann in coco["annotations"]), default=0)
    next_ann_id = max_ann_id + 1

    added = 0
    real = 0
    t0 = time.perf_counter()
    for i, (img_id, info) in enumerate(images.items()):
        src = train_dir / info["file_name"]
        dst = out_train / info["file_name"]
        if not dst.exists():
            shutil.copy2(src, dst)

        img = cv2.imread(str(src))
        H_o, W_o = img.shape[:2]
        tensor, scale = letterbox(img, teacher_size)
        tensor = tensor.to(device)

        with torch.no_grad():
            res = model.predict(tensor)[0]
        p_boxes = res["boxes"].cpu().numpy()
        p_scores = res["scores"].cpu().numpy()
        if p_boxes.size == 0:
            continue
        p_boxes = p_boxes / scale  # map back to original image coords
        keep = p_scores >= args.conf
        p_boxes = p_boxes[keep]
        p_scores = p_scores[keep]
        if p_boxes.size == 0:
            continue

        # Drop teacher boxes that already overlap a real GT (real wins)
        gts = gt_by_img.get(img_id, [])
        if gts:
            gt_xyxy = np.array(
                [[g["bbox"][0], g["bbox"][1],
                  g["bbox"][0] + g["bbox"][2], g["bbox"][1] + g["bbox"][3]]
                 for g in gts], dtype=np.float32,
            )
            ious = iou_xyxy(p_boxes, gt_xyxy).max(axis=1)
            p_boxes = p_boxes[ious < args.iou_with_gt]
            p_scores = p_scores[ious < args.iou_with_gt]

        if p_boxes.size == 0:
            continue

        # Sort by score, cap per image
        order = np.argsort(-p_scores)[: args.max_per_image]
        for j in order:
            x1, y1, x2, y2 = p_boxes[j]
            x1 = float(np.clip(x1, 0, W_o)); y1 = float(np.clip(y1, 0, H_o))
            x2 = float(np.clip(x2, 0, W_o)); y2 = float(np.clip(y2, 0, H_o))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            coco["annotations"].append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
                "_teacher": True,
            })
            next_ann_id += 1
            added += 1
        real += len(gts)

    elapsed = time.perf_counter() - t0
    print(f"\nProcessed {len(images)} train images in {elapsed:.1f}s")
    print(f"  Real GT boxes:        {real}")
    print(f"  Teacher pseudo-labels:{added}  (+{100*added/max(real,1):.0f}% over real)")
    print(f"  New total per image:  {(real+added)/max(len(images),1):.1f}")

    out_train_ann = out_train / "_annotations.coco.json"
    with open(out_train_ann, "w") as f:
        json.dump(coco, f)
    print(f"\nWrote: {out_train_ann}")


if __name__ == "__main__":
    main()
