"""
BoxVision v2: Complete model assembly.

Presets:
- tiny:  ShuffleNetV2-0.5x + 32ch FPN + 1-conv head (~162K params)
- small: ShuffleNetV2-1.0x + 48ch Ghost-FPN + 2-conv head (~500K params)
"""

import torch
import torch.nn as nn
import torchvision.ops as ops
from typing import List, Optional, Tuple

from .backbone import ShuffleNetV2Backbone
from .fpn import LightFPN
from .head import FCOSHead
from .config import ModelConfig


class BoxVision(nn.Module):
    """
    BoxVision v2: class-agnostic box detector.

    Architecture:
        ShuffleNetV2-0.5x → LightFPN (32ch) → FCOSHead (1-conv, no centerness)
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.config = config or ModelConfig()

        self.backbone = ShuffleNetV2Backbone(
            variant=self.config.backbone,
            pretrained=self.config.pretrained_backbone,
        )

        self.fpn = LightFPN(
            in_channels_list=self.backbone.get_out_channels(),
            out_channels=self.config.fpn_out_channels,
            use_ghost=self.config.fpn_use_ghost,
        )

        self.head = FCOSHead(
            in_channels=self.config.fpn_out_channels,
            num_convs=self.config.head_num_convs,
            use_depthwise=self.config.head_use_depthwise,
            use_centerness=self.config.use_centerness,
            num_levels=len(self.config.strides),
            num_classes=self.config.num_classes,
        )

        self.strides = self.config.strides
        self._grids_built = False

    def _build_grids(self, input_h: int, input_w: int, device: torch.device):
        """Pre-compute grid center points for each FPN level. Call once."""
        self._grid_points = []
        for stride in self.strides:
            h = input_h // stride
            w = input_w // stride
            x_range = torch.arange(0, w, device=device).float() * stride + stride // 2
            y_range = torch.arange(0, h, device=device).float() * stride + stride // 2
            y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
            points = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
            self._grid_points.append(points)
        self._grids_built = True

    def forward(self, x: torch.Tensor):
        """Raw head outputs only — ONNX-traceable. Same return shape in train and eval.

        Returns 4-tuple:
            objectness:   list[Tensor]  per FPN level, [B,1,H,W] logits
            bbox_reg:     list[Tensor]  per FPN level, [B,4,H,W] decoded l/t/r/b
            class_logits: list[Tensor] or None — only when num_classes > 1
            centerness:   list[Tensor] or None — only when use_centerness=True

        Use `predict()` for PyTorch inference with decode + NMS post-processing.
        """
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        return self.head(fpn_features)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """PyTorch-only inference: forward + decode + NMS. Returns list[dict].

        Each dict has:
            boxes:  [N, 4]  (x1, y1, x2, y2)
            scores: [N]     final confidence (objectness * class for multi-class)
            labels: [N]     class index (0 if num_classes == 1)
        """
        was_training = self.training
        self.eval()
        try:
            outputs = self.forward(x)
            return self._decode_and_nms(*outputs, image_size=x.shape[2:])
        finally:
            self.train(was_training)

    def decode_raw(self, objectness, bbox_reg, class_logits, centerness, image_size):
        """
        Decode head outputs to boxes/scores/labels tensors (ONNX-safe).

        No NMS, no dynamic ops. Returns all decoded predictions concatenated.

        Args:
            objectness:   list of [B,1,H,W] per FPN level
            bbox_reg:     list of [B,4,H,W] per FPN level
            class_logits: list of [B,C,H,W] per FPN level, or None
            centerness:   list of [B,1,H,W] per FPN level, or None
            image_size:   (H, W)

        Returns:
            boxes:  [B, N, 4] in (x1, y1, x2, y2) format
            scores: [B, N] confidence — for multi-class this is obj*max_class
            labels: [B, N] class index (all zeros when num_classes == 1)
        """
        device = objectness[0].device
        H, W = image_size

        if not self._grids_built:
            self._build_grids(H, W, device)

        batch_size = objectness[0].shape[0]
        batch_boxes = []
        batch_scores = []
        batch_labels = []

        for b in range(batch_size):
            level_boxes = []
            level_scores = []
            level_labels = []

            for level_idx in range(len(self.strides)):
                obj = objectness[level_idx][b, 0]
                bbox = bbox_reg[level_idx][b]
                points = self._grid_points[level_idx].to(device)

                obj_flat = obj.reshape(-1)
                bbox_flat = bbox.permute(1, 2, 0).reshape(-1, 4)
                obj_scores = torch.sigmoid(obj_flat)

                if class_logits is not None:
                    # Multi-class: combine objectness with per-class probability.
                    # Use sigmoid per class (not softmax) — multi-label friendly,
                    # and class_pred bias init expects independent logits.
                    cls = torch.sigmoid(class_logits[level_idx][b])  # [C, H, W]
                    cls_flat = cls.reshape(cls.shape[0], -1)          # [C, H*W]
                    cls_score, cls_label = cls_flat.max(dim=0)         # [H*W], [H*W]
                    scores = obj_scores * cls_score
                    labels = cls_label
                else:
                    scores = obj_scores
                    labels = torch.zeros_like(obj_flat, dtype=torch.long)

                if centerness is not None:
                    ctr = torch.sigmoid(centerness[level_idx][b, 0].reshape(-1))
                    scores = scores * ctr

                x1 = points[:, 0] - bbox_flat[:, 0]
                y1 = points[:, 1] - bbox_flat[:, 1]
                x2 = points[:, 0] + bbox_flat[:, 2]
                y2 = points[:, 1] + bbox_flat[:, 3]
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)

                level_boxes.append(boxes)
                level_scores.append(scores)
                level_labels.append(labels)

            batch_boxes.append(torch.cat(level_boxes, dim=0))
            batch_scores.append(torch.cat(level_scores, dim=0))
            batch_labels.append(torch.cat(level_labels, dim=0))

        return torch.stack(batch_boxes), torch.stack(batch_scores), torch.stack(batch_labels)

    def _decode_and_nms(
        self,
        objectness: List[torch.Tensor],
        bbox_reg: List[torch.Tensor],
        class_logits: Optional[List[torch.Tensor]],
        centerness: Optional[List[torch.Tensor]],
        image_size: Tuple[int, int],
    ) -> List[dict]:
        """Full decode + NMS (PyTorch inference only, not for ONNX).

        Class-aware NMS via torchvision.ops.batched_nms: detections with
        different class labels don't suppress each other.
        """
        all_boxes, all_scores, all_labels = self.decode_raw(
            objectness, bbox_reg, class_logits, centerness, image_size
        )
        device = all_boxes.device
        batch_size = all_boxes.shape[0]
        results = []

        for b in range(batch_size):
            boxes = all_boxes[b]
            scores = all_scores[b]
            labels = all_labels[b]

            keep = scores > self.config.objectness_threshold
            if keep.sum() == 0:
                results.append({
                    "boxes": torch.zeros(0, 4, device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                })
                continue

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            keep_nms = ops.batched_nms(boxes, scores, labels, self.config.nms_threshold)
            if len(keep_nms) > self.config.max_detections:
                keep_nms = keep_nms[:self.config.max_detections]

            results.append({
                "boxes": boxes[keep_nms],
                "scores": scores[keep_nms],
                "labels": labels[keep_nms],
            })

        return results

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_mb": total * 4 / (1024 * 1024),
        }


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters that tracks an
    exponentially weighted average of the training weights. The EMA
    model typically generalizes better than the raw training model.

    Standard technique used in YOLO, EfficientDet, etc.
    """

    def __init__(self, model: nn.Module, decay: float = 0.99, warmup_steps: int = 50):
        self.decay = decay
        self.warmup_steps = warmup_steps
        # Match the source model's device so EMA mul_/add_ ops don't fail
        # with mixed-device tensors when the source model is on CUDA.
        device = next(model.parameters()).device
        self.ema_model = BoxVision(model.config).to(device)
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.updates = 0
        self.skipped = 0

    def update(self, model: nn.Module):
        """Update EMA weights after each training step.

        Skips updates for the first `warmup_steps` training steps so the raw
        model has time to move off its initialization. After warmup, decay
        ramps from low to target via `min(decay, (1 + n) / (10 + n))` —
        standard YOLO-style warmup that prevents the early-step bias.
        """
        if self.skipped < self.warmup_steps:
            self.skipped += 1
            # Keep EMA snapped to the current model during warmup so we don't
            # carry stale init weights into the post-warmup average.
            with torch.no_grad():
                for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
                    ema_p.copy_(model_p)
                for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
                    ema_b.copy_(model_b)
            return

        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))

        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.mul_(d).add_(model_p, alpha=1 - d)

            for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
                ema_b.copy_(model_b)


def build_model(config: ModelConfig = None) -> BoxVision:
    """Factory function to create a BoxVision model."""
    return BoxVision(config)


def _verify_model():
    """Verify model shapes and parameter count."""
    config = ModelConfig(pretrained_backbone=False)
    model = build_model(config)

    params = model.count_parameters()
    print(f"BoxVision v2")
    print(f"  Backbone: {config.backbone}")
    print(f"  FPN channels: {config.fpn_out_channels}")
    print(f"  Head convs: {config.head_num_convs}")
    print(f"  Input: {config.input_size}")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Model size (FP32): {params['total_mb']:.2f} MB")

    # Training forward
    model.train()
    x = torch.randn(2, 3, *config.input_size)
    objectness, bbox_reg, centerness = model(x)

    print(f"\n  Training output shapes:")
    for i, (obj, bbox) in enumerate(zip(objectness, bbox_reg)):
        ctr_shape = "N/A" if centerness is None else str(list(centerness[i].shape))
        print(f"    P{i+3}: obj={list(obj.shape)}, bbox={list(bbox.shape)}, ctr={ctr_shape}")

    # Inference forward
    model.eval()
    with torch.no_grad():
        results = model(x)
    print(f"\n  Inference: {results[0]['boxes'].shape[0]} detections")
    print(f"\n  Verification passed!")


if __name__ == "__main__":
    _verify_model()
