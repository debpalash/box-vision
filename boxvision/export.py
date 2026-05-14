"""
ONNX export and inference for BoxVision.

Exports the trained PyTorch model to ONNX format for fast CPU inference.
Includes optional INT8 quantization for even smaller model size.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import List, Tuple, Optional

from .model import BoxVision, build_model
from .config import ModelConfig, ExportConfig


class BoxVisionONNXExporter:
    """Export BoxVision model to ONNX format."""

    def __init__(self, model: BoxVision, export_config: ExportConfig = None):
        self.model = model
        self.config = export_config or ExportConfig()

    def export(self, checkpoint_path: Optional[str] = None) -> str:
        """
        Export the model to ONNX.

        Args:
            checkpoint_path: Path to .pt checkpoint (optional, loads weights)

        Returns:
            Path to exported ONNX model
        """
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        self.model.cpu()

        # Create dummy input
        H, W = self.config.input_size
        dummy_input = torch.randn(1, 3, H, W)

        # Dynamic axes for batch size
        dynamic_axes = None
        if self.config.dynamic_axes:
            dynamic_axes = {"input": {0: "batch_size"}}
            # We need to handle multi-output dynamic axes
            # ONNX export in eval mode outputs the decoded results

        # Export
        output_path = self.config.output_path
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        print(f"Exporting BoxVision to ONNX: {output_path}")
        print(f"  Input size: {H}x{W}")
        print(f"  Opset version: {self.config.opset_version}")

        # For ONNX export, we use a wrapper that returns flat tensors
        export_model = _ONNXExportWrapper(self.model)

        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            opset_version=self.config.opset_version,
            input_names=["input"],
            output_names=["boxes", "scores", "labels"],
            dynamic_axes=dynamic_axes,
        )

        # Get file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Exported model size: {size_mb:.2f} MB")

        # Simplify if requested
        if self.config.simplify:
            try:
                import onnx
                from onnxsim import simplify

                model_onnx = onnx.load(output_path)
                model_simplified, check = simplify(model_onnx)
                if check:
                    onnx.save(model_simplified, output_path)
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"  Simplified model size: {size_mb:.2f} MB")
                else:
                    print("  Warning: ONNX simplification failed validation, keeping original")
            except ImportError:
                print("  Note: Install onnxsim for model simplification (pip install onnxsim)")

        # Quantize if requested
        if self.config.quantize_int8:
            self._quantize_int8(output_path)

        return output_path

    def _quantize_int8(self, model_path: str):
        """Apply post-training INT8 quantization."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = model_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(
                model_path,
                quantized_path,
                weight_type=QuantType.QInt8,
            )

            size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
            print(f"  INT8 quantized model: {quantized_path} ({size_mb:.2f} MB)")
        except ImportError:
            print("  Note: Install onnxruntime for INT8 quantization")


class _ONNXExportWrapper(nn.Module):
    """
    Wrapper for ONNX export that returns flat tensors.

    Uses decode_raw() which outputs [B, N, 4] boxes, [B, N] scores, [B, N] labels
    without NMS or dynamic control flow — clean ONNX graph.
    NMS is handled in the ONNX Runtime postprocess step.

    For class-agnostic models (num_classes=1) labels are all zeros.
    """

    def __init__(self, model: BoxVision):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x: torch.Tensor):
        features = self.model.backbone(x)
        fpn_features = self.model.fpn(features)
        objectness, bbox_reg, class_logits, centerness = self.model.head(fpn_features)
        boxes, scores, labels = self.model.decode_raw(
            objectness, bbox_reg, class_logits, centerness, x.shape[2:]
        )
        # Squeeze batch dim for single-image export
        return boxes[0], scores[0], labels[0]




class BoxVisionONNXInference:
    """
    Run BoxVision inference using ONNX Runtime.

    This is the primary inference class for deployment — fast CPU inference
    without PyTorch dependency (only needs onnxruntime, opencv, numpy).
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.50,
        max_detections: int = 300,
    ):
        """
        Args:
            model_path: Path to ONNX model file
            input_size: Expected input (H, W)
            confidence_threshold: Min confidence to keep a detection
            nms_threshold: NMS IoU threshold
            max_detections: Max boxes to return
        """
        import onnxruntime as ort

        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Preprocess an image for inference.

        Args:
            image: BGR image from cv2.imread

        Returns:
            input_tensor: [1, 3, H, W] normalized float32
            meta: dict with scale/padding info for box coordinate mapping
        """
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.input_size

        # Letterbox resize (maintain aspect ratio)
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # BGR -> RGB, normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std

        # HWC -> CHW -> NCHW
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "scale": scale,
            "pad_top": pad_top,
            "pad_left": pad_left,
        }

        return tensor.astype(np.float32), meta

    def postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        meta: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess ONNX outputs: map boxes back to original image coordinates.

        Args:
            boxes: [K, 4] detected boxes in model input space
            scores: [K] detection scores
            meta: Preprocessing metadata

        Returns:
            boxes: [K, 4] in original image coordinates (x1, y1, x2, y2)
            scores: [K] filtered scores
        """
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Filter by confidence
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Remove padding offset
        boxes[:, 0] -= meta["pad_left"]
        boxes[:, 1] -= meta["pad_top"]
        boxes[:, 2] -= meta["pad_left"]
        boxes[:, 3] -= meta["pad_top"]

        # Rescale to original image size
        boxes /= meta["scale"]

        # Clip to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, meta["orig_w"])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, meta["orig_h"])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, meta["orig_w"])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, meta["orig_h"])

        # Limit detections
        if len(scores) > self.max_detections:
            top_k = np.argsort(scores)[::-1][:self.max_detections]
            boxes = boxes[top_k]
            scores = scores[top_k]

        return boxes, scores

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run full detection pipeline on a single image.

        Args:
            image: BGR image from cv2.imread

        Returns:
            boxes: [K, 4] detected boxes in (x1, y1, x2, y2) format
            scores: [K] confidence scores
        """
        # Preprocess
        input_tensor, meta = self.preprocess(image)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        boxes, scores = outputs[0], outputs[1]

        # Postprocess
        boxes, scores = self.postprocess(boxes, scores, meta)

        return boxes, scores

    def detect_and_draw(
        self,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects and draw boxes on the image.

        Args:
            image: BGR image
            color: Box color (BGR)
            thickness: Line thickness

        Returns:
            drawn_image: Image with boxes drawn
            boxes: [K, 4] detected boxes
            scores: [K] scores
        """
        boxes, scores = self.detect(image)
        drawn = image.copy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, thickness)

            # Score label
            label = f"{score:.2f}"
            font_scale = 0.5
            font_thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Label background
            cv2.rectangle(drawn, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(drawn, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        return drawn, boxes, scores

    def benchmark(self, image: np.ndarray, num_runs: int = 100) -> dict:
        """
        Benchmark inference speed.

        Returns:
            Dict with avg_ms, min_ms, max_ms, fps
        """
        input_tensor, _ = self.preprocess(image)

        # Warmup
        for _ in range(10):
            self.session.run(None, {self.input_name: input_tensor})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.session.run(None, {self.input_name: input_tensor})
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        times = np.array(times)
        return {
            "avg_ms": float(np.mean(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "std_ms": float(np.std(times)),
            "fps": float(1000.0 / np.mean(times)),
        }
