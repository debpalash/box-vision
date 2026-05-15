"""
Run all three shipping ONNX variants on the same set of test images.
Reports per-image detections + timing and saves annotated visualizations.

Usage:
    python scripts/test_inference.py \
        --dataset shapes \
        --n-images 5 \
        --out runs/G-aug-300ep/inference_test
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
    color: tuple  # BGR


VARIANTS = [
    Variant("Pro  (G FP32 @ 416)",  "runs/G-aug-300ep/boxvision-G.onnx",          416, (0, 200, 0)),
    Variant("Fast (G FP32 @ 320)",  "runs/G-aug-300ep/boxvision-G-320.onnx",      320, (255, 100, 0)),
    Variant("INT8 (G INT8 @ 320)",  "runs/G-aug-300ep/boxvision-G-320-int8.onnx", 320, (0, 100, 255)),
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


def predict(sess, img_bgr, input_size: int, conf: float = 0.35, iou: float = 0.15):
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
        idx = ops.batched_nms(boxes, scores, labels, iou)[:300]
        boxes, scores, labels = boxes[idx], scores[idx], labels[idx]
        # Containment suppression: drop a smaller box whose center sits inside
        # a larger higher-scoring box. IoU-NMS misses this case when sizes
        # differ a lot (small/large ratio tanks the union).
        boxes, scores, labels = _suppress_contained(boxes, scores, labels)
        boxes = boxes.numpy() / scale
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
    return boxes, scores.numpy(), dt


def _suppress_contained(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
                          center_thresh: float = 0.4):
    """If box A's center is inside box B and B has a higher score, drop A.

    Specifically: for each pair (i, j) with score_i < score_j, compute what
    fraction of box i's area is contained inside box j. If > center_thresh
    drop box i. Handles the "tight inner + loose outer" duplicate case.
    """
    n = boxes.shape[0]
    if n <= 1:
        return boxes, scores, labels
    order = torch.argsort(scores, descending=True)
    keep_mask = torch.ones(n, dtype=torch.bool)
    for i_idx in range(n):
        i = order[i_idx].item()
        if not keep_mask[i]:
            continue
        xi1, yi1, xi2, yi2 = boxes[i]
        area_i = (xi2 - xi1) * (yi2 - yi1)
        for j_idx in range(i_idx + 1, n):
            j = order[j_idx].item()
            if not keep_mask[j]:
                continue
            xj1, yj1, xj2, yj2 = boxes[j]
            ix1 = max(xi1, xj1); iy1 = max(yi1, yj1)
            ix2 = min(xi2, xj2); iy2 = min(yi2, yj2)
            iw = max(ix2 - ix1, torch.tensor(0.0))
            ih = max(iy2 - iy1, torch.tensor(0.0))
            inter = iw * ih
            area_j = (xj2 - xj1) * (yj2 - yj1)
            min_area = min(area_i, area_j)
            if min_area <= 0:
                continue
            containment = inter / min_area
            if containment > center_thresh:
                keep_mask[j] = False
    return boxes[keep_mask], scores[keep_mask], labels[keep_mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = load_dataset(args.dataset)
    img_dir = Path(spec.val.images)
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng = random.Random(args.seed)
    picks = rng.sample(images, min(args.n_images, len(images)))

    sessions = {v.name: ort.InferenceSession(v.path, providers=["CPUExecutionProvider"])
                for v in VARIANTS}

    # Warmup
    dummy = cv2.imread(str(picks[0]))
    for v in VARIANTS:
        predict(sessions[v.name], dummy, v.input_size, args.conf)

    print(f"{'Image':<60} {'Pro detections':>18} {'Fast detections':>18} {'INT8 detections':>18}")
    print(f"{'':<60} {'mAP-anchor':>18} {'2.5×':>18} {'4.3× (low mAP)':>18}")
    print("-" * 132)

    per_variant_times = {v.name: [] for v in VARIANTS}

    for img_path in picks:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        results: dict = {}
        for v in VARIANTS:
            # Time average over 5 runs per image (warmed up)
            run_times = []
            boxes = scores = None
            for _ in range(5):
                boxes, scores, dt = predict(sessions[v.name], img_bgr, v.input_size, args.conf)
                run_times.append(dt)
            avg = sum(run_times) / len(run_times)
            results[v.name] = (boxes, scores, avg)
            per_variant_times[v.name].append(avg)

        print(f"{img_path.name:<60} "
              f"{f'{results[VARIANTS[0].name][0].shape[0]} ({results[VARIANTS[0].name][2]:.1f}ms)':>18} "
              f"{f'{results[VARIANTS[1].name][0].shape[0]} ({results[VARIANTS[1].name][2]:.1f}ms)':>18} "
              f"{f'{results[VARIANTS[2].name][0].shape[0]} ({results[VARIANTS[2].name][2]:.1f}ms)':>18}")

        # Draw side-by-side panels
        panels = []
        for v in VARIANTS:
            boxes, scores, _ = results[v.name]
            panel = img_bgr.copy()
            for j in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[j].astype(int)
                cv2.rectangle(panel, (x1, y1), (x2, y2), v.color, 2)
                cv2.putText(panel, f"{scores[j]:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, v.color, 1)
            cv2.putText(panel, v.name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2)
            cv2.putText(panel, v.name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, v.color, 1)
            panels.append(panel)
        # Pad panels to same height (they share size since we use the input image)
        combined = np.concatenate(panels, axis=1)
        out_path = out / f"{img_path.stem}_compare.jpg"
        cv2.imwrite(str(out_path), combined)

    # Summary
    print()
    print("=" * 132)
    print(f"{'Summary':<24} {'mAP@0.5':>10} {'avg ms':>10} {'vs YOLO26n':>12}")
    yolo_ms = 38.9
    map_lookup = {
        "Pro  (G FP32 @ 416)":  87.30,
        "Fast (G FP32 @ 320)":  86.48,
        "INT8 (G INT8 @ 320)":  79.05,
    }
    for v in VARIANTS:
        avg_ms = sum(per_variant_times[v.name]) / max(len(per_variant_times[v.name]), 1)
        print(f"{v.name:<24} {map_lookup[v.name]:>9.2f}% {avg_ms:>9.2f}ms {yolo_ms/avg_ms:>11.2f}×")
    print(f"\nAnnotated images saved to: {out}/")


if __name__ == "__main__":
    main()
