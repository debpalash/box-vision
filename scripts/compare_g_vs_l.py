"""Side-by-side gallery comparing G vs L (the all-fixes variant) on val images."""

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
    Variant("G (P2+aug)",         "runs/G-aug-300ep/boxvision-G.onnx",         416, (0, 200, 0)),
    Variant("L (G+all-fixes)",    "runs/L-allfixes-300ep/boxvision-L-416.onnx", 416, (255, 100, 0)),
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


def predict(sess, img, input_size, conf=0.35, iou=0.5):
    tensor, scale = letterbox(img, input_size)
    t0 = time.perf_counter()
    outs = sess.run(None, {"input": tensor[None, ...]})
    dt = (time.perf_counter() - t0) * 1000
    boxes = torch.from_numpy(outs[0])
    scores = torch.from_numpy(outs[1])
    labels = torch.from_numpy(outs[2])
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
    import random
    spec = load_dataset("shapes")
    img_dir = Path(spec.val.images)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng = random.Random(42)
    picks = rng.sample(images, 5)

    out = Path("runs/L-allfixes-300ep/gallery_g_vs_l")
    out.mkdir(parents=True, exist_ok=True)

    sessions = {v.name: ort.InferenceSession(v.path, providers=["CPUExecutionProvider"]) for v in VARIANTS}

    print(f"{'Image':<50} {'G boxes':>10} {'L boxes':>10}")
    print("-" * 75)
    for p in picks:
        img = cv2.imread(str(p))
        panels = []
        counts = []
        for v in VARIANTS:
            boxes, scores, _ = predict(sessions[v.name], img, v.input_size, conf=0.35, iou=0.5)
            counts.append(boxes.shape[0])
            panel = img.copy()
            for j in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[j].astype(int)
                cv2.rectangle(panel, (x1, y1), (x2, y2), v.color, 2)
                cv2.putText(panel, f"{scores[j]:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, v.color, 1)
            cv2.putText(panel, v.name, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(panel, v.name, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, v.color, 1)
            panels.append(panel)
        print(f"{p.name:<50} {counts[0]:>10} {counts[1]:>10}")
        combined = np.concatenate(panels, axis=1)
        cv2.imwrite(str(out / f"{p.stem}_gvsl.jpg"), combined)

    print(f"\nGallery: {out}/")


if __name__ == "__main__":
    main()
