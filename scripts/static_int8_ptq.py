"""
Static INT8 post-training quantization for a BoxVision ONNX model.

Static PTQ inserts QuantizeLinear/DequantizeLinear ops around convs and
"freezes" activation ranges using a small calibration set. Lower runtime
overhead than dynamic quantization → real CPU speedup.

Usage:
    python scripts/static_int8_ptq.py \
        --onnx runs/G-aug-300ep/boxvision-G-320.onnx \
        --dataset shapes \
        --input-size 320 \
        --calibration-samples 64 \
        --out runs/G-aug-300ep/boxvision-G-320-int8.onnx
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader,
    CalibrationMethod,
)

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
    return rgb.transpose(2, 0, 1).astype(np.float32)


class CalibReader(CalibrationDataReader):
    """Feeds calibration tensors to the ONNX quantizer."""

    def __init__(self, image_paths, input_size: int):
        self.paths = list(image_paths)
        self.size = input_size
        self.idx = 0
        self.input_name = "input"

    def get_next(self):
        if self.idx >= len(self.paths):
            return None
        img = cv2.imread(str(self.paths[self.idx]))
        self.idx += 1
        if img is None:
            return self.get_next()
        tensor = letterbox(img, self.size)
        return {self.input_name: tensor[None, ...]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--calibration-samples", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = load_dataset(args.dataset)
    train_dir = Path(spec.train.images)
    samples = sorted(p for p in train_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    samples = samples[: args.calibration_samples]
    print(f"Calibrating on {len(samples)} train images")

    reader = CalibReader(samples, args.input_size)
    t0 = time.perf_counter()
    quantize_static(
        model_input=args.onnx,
        model_output=args.out,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,            # QDQ format is the modern choice
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"Quantized in {time.perf_counter() - t0:.1f}s")

    import os
    fp_kb = os.path.getsize(args.onnx) / 1024
    q_kb = os.path.getsize(args.out) / 1024
    print(f"FP32: {fp_kb:.1f} KB   INT8: {q_kb:.1f} KB")

    # Benchmark FP32 vs INT8
    inp = np.random.randn(1, 3, args.input_size, args.input_size).astype(np.float32)
    for path, label in [(args.onnx, "FP32"), (args.out, "INT8")]:
        for nthreads in (1, 4):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = nthreads
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            for _ in range(10):
                sess.run(None, {"input": inp})
            N = 100
            t0 = time.perf_counter()
            for _ in range(N):
                sess.run(None, {"input": inp})
            dt = (time.perf_counter() - t0) / N * 1000
            print(f"  {label}  threads={nthreads}  {dt:.2f}ms  ({1000/dt:.0f} FPS)")


if __name__ == "__main__":
    main()
