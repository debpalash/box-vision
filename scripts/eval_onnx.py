"""
Evaluate an ONNX model (FP32 or INT8) on a val set, reporting COCO mAP.

We don't have NMS in the exported graph (it's done in Python), so this script
runs the ONNX, filters by score threshold, runs class-aware NMS, then feeds
predictions to pycocotools.
"""

import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.ops as ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from boxvision.registry import load_dataset


def letterbox(img, size):
    H_o, W_o = img.shape[:2]
    s = min(size / H_o, size / W_o)
    nh, nw = int(round(H_o * s)), int(round(W_o * s))
    r = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = r
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return rgb.transpose(2, 0, 1).astype(np.float32), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    spec = load_dataset(args.dataset)
    img_dir = Path(spec.val.images)
    with open(spec.val.annotations) as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}

    coco_pred = []
    coco_gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}
    total_ms = 0.0

    for img_id, info in images.items():
        img = cv2.imread(str(img_dir / info["file_name"]))
        if img is None:
            continue
        H_o, W_o = img.shape[:2]
        tensor, scale = letterbox(img, args.input_size)
        coco_gt["images"].append({"id": img_id, "height": H_o, "width": W_o})

        t0 = time.perf_counter()
        outputs = sess.run(None, {"input": tensor[None, ...]})
        total_ms += (time.perf_counter() - t0) * 1000

        # The export outputs (boxes, scores, labels) in network coords.
        boxes = torch.from_numpy(outputs[0])
        scores = torch.from_numpy(outputs[1])
        labels = torch.from_numpy(outputs[2])

        keep = scores > args.conf
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if boxes.numel() > 0:
            idx = ops.batched_nms(boxes, scores, labels, args.nms_iou)[:300]
            boxes, scores, labels = boxes[idx], scores[idx], labels[idx]
            boxes = boxes.numpy() / scale  # map back to original
            scores = scores.numpy()
            for j in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[j]
                coco_pred.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(scores[j]),
                })

    # Bring over GTs (renormalize category to 1)
    for ann in coco["annotations"]:
        coco_gt["annotations"].append({**ann, "category_id": 1})

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_gt:
        json.dump(coco_gt, f_gt); gt_path = f_gt.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_pr:
        json.dump(coco_pred, f_pr); pr_path = f_pr.name

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        gt = COCO(gt_path); dt = gt.loadRes(pr_path) if coco_pred else None
        if dt is None:
            print("no predictions"); return
        ev = COCOeval(gt, dt, iouType="bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()

    print(f"  {args.onnx}")
    print(f"  mAP@0.5:        {ev.stats[1]*100:6.2f}%")
    print(f"  mAP@0.5:0.95:   {ev.stats[0]*100:6.2f}%")
    print(f"  avg latency:    {total_ms/max(len(images),1):.2f} ms/image")
    print(f"  predictions:    {len(coco_pred)}")


if __name__ == "__main__":
    main()
