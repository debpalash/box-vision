"""
Road-signs inference gallery: teacher (small+P2+aug @ 416) vs Tiny student.
Mirrors test_inference.py but with road-signs variants.
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.ops as ops

from boxvision.registry import load_dataset


@dataclass
class Variant:
    name: str
    path: str
    input_size: int
    color: tuple


VARIANTS = [
    Variant("Teacher (small+P2 @ 416)",  "runs/rs-teacher-416-200ep/boxvision-rs-teacher.onnx", 416, (0, 200, 0)),
    Variant("Tiny KD (FP32 @ 320)",      "runs/rs-tiny-kd-200ep/boxvision-rs-tiny-p2.onnx",      320, (255, 100, 0)),
    Variant("Tiny KD (INT8 @ 320)",      "runs/rs-tiny-kd-200ep/boxvision-rs-tiny-p2-int8.onnx", 320, (255, 0, 200)),
]


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


def predict(sess, img_bgr, input_size: int, conf: float = 0.35, iou: float = 0.5):
    tensor, scale = letterbox(img_bgr, input_size)
    t0 = time.perf_counter()
    outputs = sess.run(None, {"input": tensor[None, ...]})
    dt = (time.perf_counter() - t0) * 1000
    boxes = torch.from_numpy(outputs[0])
    scores = torch.from_numpy(outputs[1])
    labels = torch.from_numpy(outputs[2])
    keep = scores > conf
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.numel() > 0:
        idx = ops.batched_nms(boxes, scores, labels, iou)[:200]
        boxes = (boxes[idx].numpy() / scale)
        scores = scores[idx].numpy()
    else:
        boxes = np.zeros((0, 4))
        scores = np.zeros((0,))
    return boxes, scores, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--out", default="runs/rs-tiny-kd-200ep/gallery_compare")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_dataset("road-signs")
    images = sorted(p for p in Path(spec.val.images).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    picks = random.Random(42).sample(images, args.n_images)

    sessions = {v.name: ort.InferenceSession(v.path, providers=["CPUExecutionProvider"])
                for v in VARIANTS}

    for img_path in picks:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        panels = []
        for v in VARIANTS:
            boxes, scores, dt = predict(sessions[v.name], img_bgr, v.input_size, args.conf)
            panel = img_bgr.copy()
            for j in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[j].astype(int)
                cv2.rectangle(panel, (x1, y1), (x2, y2), v.color, 2)
                cv2.putText(panel, f"{scores[j]:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, v.color, 1)
            cv2.putText(panel, v.name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(panel, v.name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, v.color, 1)
            panels.append(panel)
        combined = np.concatenate(panels, axis=1)
        cv2.imwrite(str(out / f"{img_path.stem}_compare.jpg"), combined)

    print(f"Saved {len(picks)} panels to {out}/")


if __name__ == "__main__":
    main()
