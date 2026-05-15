"""Diagnose duplicate-box behavior on a single image."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.ops as ops


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
    ap.add_argument("--onnx", default="runs/G-aug-300ep/boxvision-G-320.onnx")
    ap.add_argument("--image", required=True)
    ap.add_argument("--input-size", type=int, default=320)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    img = cv2.imread(args.image)
    tensor, scale = letterbox(img, args.input_size)
    boxes_raw, scores_raw, labels_raw = sess.run(None, {"input": tensor[None, ...]})
    print(f"Raw output count: {scores_raw.shape[0]}")
    print(f"Score distribution: min={scores_raw.min():.4f} max={scores_raw.max():.4f}")
    print(f"  >0.05: {(scores_raw > 0.05).sum()}")
    print(f"  >0.1:  {(scores_raw > 0.1).sum()}")
    print(f"  >0.2:  {(scores_raw > 0.2).sum()}")
    print(f"  >0.35: {(scores_raw > 0.35).sum()}")
    print(f"  >0.5:  {(scores_raw > 0.5).sum()}")
    print()

    # Try multiple conf + NMS combos
    boxes_t = torch.from_numpy(boxes_raw)
    scores_t = torch.from_numpy(scores_raw)
    labels_t = torch.from_numpy(labels_raw)
    for conf in [0.20, 0.35, 0.50]:
        keep = scores_t > conf
        b, s, l = boxes_t[keep], scores_t[keep], labels_t[keep]
        print(f"conf>{conf}: {b.shape[0]} preds")
        for nms_iou in [0.5, 0.3, 0.2]:
            idx = ops.nms(b, s, nms_iou)  # class-agnostic NMS (single class)
            print(f"   NMS@{nms_iou}: {idx.numel()} kept")

    # For default conf=0.35 nms=0.5, show pairwise IoU of survivors
    keep = scores_t > 0.35
    b, s, _ = boxes_t[keep], scores_t[keep], labels_t[keep]
    idx = ops.nms(b, s, 0.5)
    survivors = b[idx]
    print(f"\n{idx.numel()} boxes after NMS @ 0.5 — pairwise IoU matrix:")
    ious = ops.box_iou(survivors, survivors).numpy()
    np.set_printoptions(precision=2, suppress=True)
    print(ious)
    # Highlight near-duplicate pairs
    high_ovlp = []
    for i in range(ious.shape[0]):
        for j in range(i + 1, ious.shape[1]):
            if ious[i, j] > 0.3:
                high_ovlp.append((i, j, float(ious[i, j])))
    if high_ovlp:
        print(f"\nPairs with IoU > 0.3 (potential duplicates):")
        for i, j, v in high_ovlp:
            print(f"  box {i} vs box {j}: IoU={v:.3f}   "
                  f"scores ({s[idx[i]]:.2f}, {s[idx[j]]:.2f})")
    else:
        print("\nNo pairs exceed IoU=0.3 — boxes are distinct.")


if __name__ == "__main__":
    main()
