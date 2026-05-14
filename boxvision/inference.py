"""
PyTorch-based inference for BoxVision (development/testing).

Use this during development when you don't have an ONNX export yet.
For production deployment, use BoxVisionONNXInference from export.py.
"""

import time
import cv2
import numpy as np
import torch
from typing import Tuple, Optional

from .model import BoxVision, build_model
from .config import ModelConfig


class BoxVisionDetector:
    """
    PyTorch-based BoxVision detector.

    For development and testing — loads a .pt checkpoint and runs
    inference using PyTorch directly.
    """

    # ImageNet normalization
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        input_size: Tuple[int, int] = (320, 320),
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.50,
    ):
        self.device = torch.device(device)
        self.input_size = input_size

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Reconstruct model config from checkpoint
        model_config = checkpoint.get("model_config", ModelConfig())
        model_config.objectness_threshold = confidence_threshold
        model_config.nms_threshold = nms_threshold
        model_config.pretrained_backbone = False

        self.model = build_model(model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Loaded BoxVision from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', '?')}")
        print(f"  Device: {self.device}")

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, dict]:
        """Preprocess image with letterbox resize and normalization."""
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.input_size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # Normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.MEAN) / self.STD

        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        meta = {
            "orig_h": orig_h, "orig_w": orig_w,
            "scale": scale, "pad_top": pad_top, "pad_left": pad_left,
        }
        return tensor, meta

    def postprocess(self, boxes: np.ndarray, scores: np.ndarray, meta: dict):
        """Map boxes back to original image coordinates."""
        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)

        boxes[:, 0] -= meta["pad_left"]
        boxes[:, 1] -= meta["pad_top"]
        boxes[:, 2] -= meta["pad_left"]
        boxes[:, 3] -= meta["pad_top"]
        boxes /= meta["scale"]

        boxes[:, 0] = np.clip(boxes[:, 0], 0, meta["orig_w"])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, meta["orig_h"])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, meta["orig_w"])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, meta["orig_h"])

        return boxes, scores

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run detection on a BGR image."""
        tensor, meta = self.preprocess(image)
        results = self.model(tensor)

        if len(results) == 0 or results[0]["boxes"].shape[0] == 0:
            return np.zeros((0, 4)), np.zeros(0)

        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()

        return self.postprocess(boxes, scores, meta)

    def detect_and_draw(self, image: np.ndarray, color=(0, 255, 0), thickness=2):
        """Detect and visualize boxes on the image."""
        boxes, scores = self.detect(image)
        drawn = image.copy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, thickness)
            label = f"{score:.2f}"
            cv2.putText(drawn, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return drawn, boxes, scores
