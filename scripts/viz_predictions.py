"""
Render champion predictions vs ground truth on a spread of val images.

Mirrors eval_onnx.py's pipeline (letterbox -> ONNX -> conf filter -> NMS) so the
pictures show exactly what the measured numbers measure. GT green, preds red.

Usage: viz_predictions.py --onnx X --dataset hcap [--out DIR] [--n 6] [--conf 0.35]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.ops as ops

from boxvision.registry import load_dataset
from scripts.eval_onnx import letterbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="/home/ubuntu/viz")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    spec = load_dataset(args.dataset)
    img_dir = Path(spec.val.images)
    with open(spec.val.annotations) as f:
        coco = json.load(f)
    gt_by_img = {}
    for ann in coco["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann["bbox"])

    # Spread by GT count: sparsest, median, most crowded.
    ranked = sorted(coco["images"], key=lambda im: len(gt_by_img.get(im["id"], [])))
    k = args.n // 3
    picks = ranked[:k] + ranked[len(ranked) // 2 - k // 2: len(ranked) // 2 - k // 2 + k] + ranked[-k:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for info in picks:
        img = cv2.imread(str(img_dir / info["file_name"]))
        if img is None:
            continue
        tensor, scale = letterbox(img, args.input_size)
        boxes_r, scores_r, labels_r = sess.run(None, {"input": tensor[None, ...]})
        boxes = torch.from_numpy(boxes_r)
        scores = torch.from_numpy(scores_r)
        keep = scores > args.conf
        boxes, scores = boxes[keep], scores[keep]
        n_pred = 0
        if boxes.numel() > 0:
            idx = ops.batched_nms(boxes, scores, torch.from_numpy(labels_r)[keep], args.nms_iou)[:300]
            boxes = boxes[idx].numpy() / scale
            scores = scores[idx].numpy()
            n_pred = boxes.shape[0]

        for x, y, w, h in gt_by_img.get(info["id"], []):
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0), 2)
        for j in range(n_pred):
            x1, y1, x2, y2 = boxes[j].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{scores[j]:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        n_gt = len(gt_by_img.get(info["id"], []))
        name = f"gt{n_gt:02d}_pred{n_pred:02d}_{Path(info['file_name']).stem[:24]}.jpg"
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"{name}: {n_gt} gt / {n_pred} pred")


if __name__ == "__main__":
    main()
