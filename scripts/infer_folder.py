"""
Batch inference on a folder of images with an exported BoxVision ONNX model.

No ground truth needed. For each image: letterbox (matches the trained/eval
pipeline in eval_onnx.py) -> ONNX -> conf filter -> NMS -> draw boxes. Saves an
annotated image per input plus a single detections.json.

Usage: infer_folder.py --onnx X --images DIR --out DIR [--conf 0.35] [--nms-iou 0.5]
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

from scripts.eval_onnx import letterbox

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    manifest = {}
    total_dets = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        tensor, scale = letterbox(img, args.input_size)
        boxes_r, scores_r, labels_r = sess.run(None, {in_name: tensor[None, ...]})
        boxes = torch.from_numpy(boxes_r)
        scores = torch.from_numpy(scores_r)
        keep = scores > args.conf
        boxes, scores = boxes[keep], scores[keep]
        dets = []
        if boxes.numel() > 0:
            idx = ops.batched_nms(boxes, scores, torch.from_numpy(labels_r)[keep],
                                  args.nms_iou)[:300]
            boxes = (boxes[idx].numpy() / scale)
            scores = scores[idx].numpy()
            for j in range(boxes.shape[0]):
                x1, y1, x2, y2 = [float(v) for v in boxes[j]]
                dets.append({"box_xyxy": [round(x1, 1), round(y1, 1),
                                          round(x2, 1), round(y2, 1)],
                             "score": round(float(scores[j]), 4)})
                cx1, cy1, cx2, cy2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
                cv2.putText(img, f"{scores[j]:.2f}", (cx1, max(12, cy1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"det_{p.name}"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        manifest[p.name] = {"num_detections": len(dets), "detections": dets}
        total_dets += len(dets)
        print(f"{p.name}: {len(dets)} boxes")

    (out_dir / "detections.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} images, {total_dets} detections -> {out_dir}")


if __name__ == "__main__":
    main()
