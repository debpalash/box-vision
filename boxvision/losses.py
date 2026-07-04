"""
Loss functions v2 for BoxVision.

v2 changes:
- Task-Aligned Assigner (TAL) replaces FCOS centerness-based assignment
- Removed centerness loss
- Cleaner GIoU computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from typing import List, Optional, Tuple


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss (VFL) for objectness with soft IoU targets.

    Properly handles continuous targets in [0, 1] — unlike standard Focal Loss
    which assumes binary targets.

    For negatives (target == 0):
        loss = -pred^gamma * log(1 - pred)
    For positives (target > 0):
        loss = -target * (target - pred)^gamma * log(pred)

    Reference: VarifocalNet (Zhang et al., 2021)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()

        # Separate positive and negative masks
        pos_mask = target > 0
        neg_mask = ~pos_mask

        # Negative loss: weighted by pred^gamma (hard negative mining)
        # Floor at 0.01 to prevent gradient death when pred ≈ 0.01 at init
        neg_weight = torch.clamp(pred_sigmoid.pow(self.gamma), min=0.01)
        neg_loss = -neg_weight * F.logsigmoid(-pred) * neg_mask.float()

        # Positive loss: weighted by target * |target - pred|^gamma
        pos_weight = target * (target - pred_sigmoid).abs().pow(self.gamma)
        pos_loss = -pos_weight * F.logsigmoid(pred) * pos_mask.float()

        loss = self.alpha * pos_loss + (1 - self.alpha) * neg_loss
        num_pos = max(pos_mask.sum().item(), 1)
        return loss.sum() / num_pos


class GIoULoss(nn.Module):
    """
    Generalized IoU loss for bbox regression.

    GIoU = IoU - |C \\\\ (A union B)| / |C|
    Loss = 1 - GIoU
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[0] == 0:
            return pred.sum() * 0.0

        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + 1e-7)

        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
        return (1 - giou).mean()


def _compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2)
        boxes2: [M, 4] in (x1, y1, x2, y2)

    Returns:
        iou: [N, M]
    """
    N, M = boxes1.shape[0], boxes2.shape[0]
    a = boxes1[:, None, :].expand(N, M, 4)
    b = boxes2[None, :, :].expand(N, M, 4)

    inter_x1 = torch.max(a[..., 0], b[..., 0])
    inter_y1 = torch.max(a[..., 1], b[..., 1])
    inter_x2 = torch.min(a[..., 2], b[..., 2])
    inter_y2 = torch.min(a[..., 3], b[..., 3])
    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    return inter / (area_a + area_b - inter + 1e-7)


class TaskAlignedAssigner:
    """
    Task-Aligned Assigner (TAL) with optional soft labels (DSLA-style).

    Standard TAL: picks best anchor-point per GT using alignment metric.
    Soft labels: positive targets are IoU values instead of hard 1.0.
    This gives the model a richer training signal — the key insight from
    NanoDet-Plus that provided +7 mAP improvement.

    alignment_metric = objectness^alpha * iou^beta
    """

    # FCOS-style per-FPN-level size ranges (max box dimension).
    # An overlapping schedule (a box can train two adjacent levels) is more
    # forgiving than a hard cut and gives stronger training signal.
    DEFAULT_STRIDE_RANGES = {
        4:  (0,   64),
        8:  (32,  128),
        16: (64,  256),
        32: (128, 1e6),
    }

    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0,
                 use_soft_labels: bool = True,
                 # FCOS-style per-FPN-level matching helped mAP@0.5 (+1.1) but
                 # hurt strict-IoU mAP (-14.5) and visual quality on the small
                 # shapes dataset. Kept in code as opt-in; default is off.
                 use_level_matching: bool = False,
                 stride_ranges: Optional[dict] = None):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.use_soft_labels = use_soft_labels
        self.use_level_matching = use_level_matching
        self.stride_ranges = stride_ranges or self.DEFAULT_STRIDE_RANGES

    @torch.no_grad()
    def assign(
        self,
        obj_scores: torch.Tensor,    # [num_points] objectness scores (sigmoid)
        pred_boxes: torch.Tensor,     # [num_points, 4] predicted boxes (xyxy)
        gt_boxes: torch.Tensor,       # [num_gt, 4] ground truth boxes (xyxy)
        points: torch.Tensor,         # [num_points, 2] grid center points (x, y)
        point_strides: Optional[torch.Tensor] = None,  # [num_points] stride per point
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assign GT boxes to prediction points using task-aligned metric.

        Returns:
            labels: [num_points] soft/hard labels (IoU or 1.0 for positives)
            bbox_targets: [num_points, 4] target boxes for positives (xyxy)
            assign_metrics: [num_points] alignment metric for positive weighting
            assigned_gt: [num_points] long tensor of GT index per point (only valid where labels > 0)
        """
        num_points = points.shape[0]
        num_gt = gt_boxes.shape[0]
        device = points.device

        if num_gt == 0:
            return (
                torch.zeros(num_points, device=device),
                torch.zeros(num_points, 4, device=device),
                torch.zeros(num_points, device=device),
                torch.zeros(num_points, dtype=torch.long, device=device),
            )

        # 1. Filter: only consider points inside GT boxes
        px = points[:, 0:1]  # [N, 1]
        py = points[:, 1:2]  # [N, 1]
        gt_x1 = gt_boxes[:, 0:1].T  # [1, M]
        gt_y1 = gt_boxes[:, 1:2].T
        gt_x2 = gt_boxes[:, 2:3].T
        gt_y2 = gt_boxes[:, 3:4].T

        inside_mask = (px >= gt_x1) & (px <= gt_x2) & (py >= gt_y1) & (py <= gt_y2)  # [N, M]

        # 1b. Per-FPN-level size matching: only let appropriate levels claim
        # each GT based on the GT's max(w, h). Skipped if no stride info is
        # provided or feature is disabled.
        if self.use_level_matching and point_strides is not None:
            gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
            gt_size = torch.maximum(gt_w, gt_h)  # [M]
            # Build a [N, M] mask: True if point's stride covers this GT's size.
            level_mask = torch.zeros_like(inside_mask)
            for stride, (lo, hi) in self.stride_ranges.items():
                level_pts = (point_strides == stride)
                # mask[N, M] for this stride: True where GT fits in [lo, hi]
                fits = (gt_size >= lo) & (gt_size < hi)
                level_mask |= level_pts[:, None] & fits[None, :]
            inside_mask = inside_mask & level_mask

        # 2. Compute alignment metric
        iou = _compute_iou(pred_boxes, gt_boxes)  # [N, M]
        obj_expanded = obj_scores[:, None].expand(-1, num_gt)  # [N, M]
        alignment = obj_expanded.pow(self.alpha) * iou.pow(self.beta)
        alignment = alignment * inside_mask.float()

        # 3. Select top-k per GT box
        topk_mask = torch.zeros_like(alignment, dtype=torch.bool)  # [N, M]
        topk_k = min(self.topk, num_points)

        for gt_idx in range(num_gt):
            gt_alignment = alignment[:, gt_idx]
            if gt_alignment.sum() == 0:
                continue
            _, topk_indices = gt_alignment.topk(topk_k, dim=0)
            topk_mask[topk_indices, gt_idx] = True

        candidate_mask = topk_mask & inside_mask  # [N, M]

        # 4. Resolve conflicts: each point assigned to at most 1 GT (highest IoU)
        candidate_iou = iou * candidate_mask.float()
        max_iou, assigned_gt = candidate_iou.max(dim=1)  # [N]
        is_positive = max_iou > 0

        # 5. Labels: soft (IoU-based) or hard (binary)
        if self.use_soft_labels:
            # DSLA-style: positive targets = IoU with assigned GT
            # This gives the model richer supervision — a point with IoU 0.9
            # should have higher objectness than one with IoU 0.3
            labels = max_iou  # Soft: [0, 1] range based on IoU
        else:
            labels = is_positive.float()  # Hard: 0 or 1

        # Gather targets
        bbox_targets = gt_boxes[assigned_gt]  # [N, 4]
        bbox_targets = bbox_targets * is_positive[:, None].float()  # Zero out negatives

        # Alignment metric for loss weighting
        assign_metrics = alignment[torch.arange(num_points, device=device), assigned_gt]
        assign_metrics = assign_metrics * is_positive.float()

        return labels, bbox_targets, assign_metrics, assigned_gt


class BoxVisionLoss(nn.Module):
    """
    Combined loss for BoxVision v2.

    Uses TAL assigner for positive sample selection.
    No centerness loss — cleaner training signal.
    """

    def __init__(
        self,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        objectness_weight: float = 1.0,
        bbox_weight: float = 2.0,
        class_weight: float = 1.0,
        strides: List[int] = None,
        tal_topk: int = 10,
        tal_alpha: float = 0.5,
        tal_beta: float = 6.0,
        use_soft_labels: bool = True,
        aux_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.obj_loss_fn = VarifocalLoss(focal_alpha, focal_gamma)
        self.giou_loss = GIoULoss()
        self.objectness_weight = objectness_weight
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_loss_weight = aux_loss_weight
        self.strides = strides or [8, 16, 32]

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, alpha=tal_alpha, beta=tal_beta,
            use_soft_labels=use_soft_labels,
        )

    def _get_points(self, featmap_size: Tuple[int, int], stride: int,
                    device: torch.device) -> torch.Tensor:
        """Generate grid center points for one FPN level."""
        h, w = featmap_size
        x_range = torch.arange(0, w, device=device).float() * stride + stride // 2
        y_range = torch.arange(0, h, device=device).float() * stride + stride // 2
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)

    def _ltrb_to_xyxy(self, points: torch.Tensor, ltrb: torch.Tensor) -> torch.Tensor:
        """Convert (l, t, r, b) distances from points to (x1, y1, x2, y2)."""
        x1 = points[:, 0] - ltrb[:, 0]
        y1 = points[:, 1] - ltrb[:, 1]
        x2 = points[:, 0] + ltrb[:, 2]
        y2 = points[:, 1] + ltrb[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _flatten_image(
        self,
        objectness_preds: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        class_logits_preds: Optional[List[torch.Tensor]],
        b: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Flatten one image's per-level predictions into per-point tensors.

        Returns:
            cat_obj:          [N] objectness logits
            cat_bbox_decoded: [N, 4] decoded boxes (xyxy)
            cat_class:        [N, C] class logits, or None
            cat_points:       [N, 2] grid center points
            cat_strides:      [N] stride per point
        """
        multi_class = class_logits_preds is not None
        num_classes = class_logits_preds[0].shape[1] if multi_class else 0

        all_obj = []
        all_bbox_decoded = []
        all_points = []
        all_strides = []
        all_class = [] if multi_class else None

        for level_idx in range(len(objectness_preds)):
            stride = self.strides[level_idx]
            obj = objectness_preds[level_idx][b, 0]   # [H, W]
            bbox = bbox_preds[level_idx][b]            # [4, H, W]
            H, W = obj.shape

            points = self._get_points((H, W), stride, device)
            bbox_flat = bbox.permute(1, 2, 0).reshape(-1, 4)

            all_obj.append(obj.reshape(-1))
            all_bbox_decoded.append(self._ltrb_to_xyxy(points, bbox_flat))
            all_points.append(points)
            all_strides.append(torch.full((points.shape[0],), stride,
                                          dtype=torch.long, device=device))

            if multi_class:
                # [C, H, W] → [H*W, C]
                cls = class_logits_preds[level_idx][b].permute(1, 2, 0).reshape(-1, num_classes)
                all_class.append(cls)

        return (
            torch.cat(all_obj, dim=0),
            torch.cat(all_bbox_decoded, dim=0),
            torch.cat(all_class, dim=0) if multi_class else None,
            torch.cat(all_points, dim=0),
            torch.cat(all_strides, dim=0),
        )

    def _image_losses(
        self,
        cat_obj: torch.Tensor,
        cat_bbox_decoded: torch.Tensor,
        cat_class: Optional[torch.Tensor],
        labels: torch.Tensor,
        bbox_targets: torch.Tensor,
        assigned_gt: torch.Tensor,
        gt_labels: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Score one head's flattened predictions against a fixed assignment."""
        obj_loss = self.obj_loss_fn(cat_obj, labels)
        bbox_loss = torch.tensor(0.0, device=device)
        class_loss = torch.tensor(0.0, device=device)

        pos_mask = labels > 0
        num_pos = int(pos_mask.sum().item())

        if num_pos > 0:
            bbox_loss = self.giou_loss(cat_bbox_decoded[pos_mask], bbox_targets[pos_mask])

            if cat_class is not None and gt_labels is not None:
                # Class index for each positive: gt_labels[assigned_gt[positive_idx]]
                pos_gt_idx = assigned_gt[pos_mask]
                pos_class_target_idx = gt_labels[pos_gt_idx]  # [P]
                pos_class_logits = cat_class[pos_mask]         # [P, C]

                # Hard one-hot target — soft IoU weighting collapses at init
                # (IoU ≈ 0 → target ≈ 0 → no gradient signal for class). The
                # objectness branch already encodes "how confident this is an
                # object" via soft labels; class is just "which one of N".
                target_onehot = torch.zeros_like(pos_class_logits)
                target_onehot[torch.arange(num_pos, device=device), pos_class_target_idx] = 1.0

                # Sigmoid focal loss (per-class binary). Normalize by num_pos
                # so the magnitude matches obj/bbox losses.
                cls_loss = sigmoid_focal_loss(
                    pos_class_logits, target_onehot,
                    alpha=self.focal_alpha, gamma=self.focal_gamma,
                )
                class_loss = cls_loss.sum() / max(num_pos, 1)

        return obj_loss, bbox_loss, class_loss, num_pos

    def _combine(self, obj_loss: torch.Tensor, bbox_loss: torch.Tensor,
                 class_loss: torch.Tensor, multi_class: bool) -> torch.Tensor:
        total = self.objectness_weight * obj_loss + self.bbox_weight * bbox_loss
        if multi_class:
            total = total + self.class_weight * class_loss
        return total

    def forward(
        self,
        objectness_preds: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        class_logits_preds: Optional[List[torch.Tensor]],
        centerness_preds: Optional[List[torch.Tensor]],
        gt_boxes_batch: List[torch.Tensor],
        gt_labels_batch: Optional[List[torch.Tensor]] = None,
        aux_preds: Optional[tuple] = None,
    ) -> dict:
        """
        Compute losses using TAL assignment.

        Args:
            objectness_preds: List of [B, 1, Hi, Wi] per level
            bbox_preds: List of [B, 4, Hi, Wi] per level
            class_logits_preds: List of [B, C, Hi, Wi] per level, or None (Step 3 adds the loss term)
            centerness_preds: Ignored in v2 (kept for API compat)
            gt_boxes_batch: List of [Mi, 4] GT boxes per image (x1,y1,x2,y2)
            gt_labels_batch: List of [Mi] class indices per image, or None (Step 3 wires it up)
            aux_preds: AGM aux head outputs (objectness, bbox_reg, class_logits,
                centerness), or None. When given, the TAL assignment is computed
                from the aux predictions and BOTH heads train against it;
                total = light_loss + aux_loss_weight * aux_loss.
        """
        device = objectness_preds[0].device
        batch_size = objectness_preds[0].shape[0]
        multi_class = class_logits_preds is not None

        total_obj_loss = torch.tensor(0.0, device=device)
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_class_loss = torch.tensor(0.0, device=device)
        total_aux_loss = torch.tensor(0.0, device=device)
        total_positives = 0

        for b in range(batch_size):
            cat_obj, cat_bbox_decoded, cat_class, cat_points, cat_strides = \
                self._flatten_image(objectness_preds, bbox_preds, class_logits_preds, b, device)

            if aux_preds is not None:
                aux_obj, aux_bbox_decoded, aux_class, _, _ = \
                    self._flatten_image(aux_preds[0], aux_preds[1], aux_preds[2], b, device)
                # AGM: assignment comes from the stronger aux head. Detached —
                # no gradient flows through the assignment itself.
                assign_scores = torch.sigmoid(aux_obj.detach())
                assign_boxes = aux_bbox_decoded.detach()
            else:
                assign_scores = torch.sigmoid(cat_obj)
                assign_boxes = cat_bbox_decoded

            # TAL assignment
            gt_boxes = gt_boxes_batch[b].to(device)
            labels, bbox_targets, assign_metrics, assigned_gt = self.assigner.assign(
                assign_scores, assign_boxes, gt_boxes, cat_points,
                point_strides=cat_strides,
            )

            gt_labels = None
            if multi_class and gt_labels_batch is not None:
                gt_labels = gt_labels_batch[b].to(device)

            obj_loss, bbox_loss, class_loss, num_pos = self._image_losses(
                cat_obj, cat_bbox_decoded, cat_class,
                labels, bbox_targets, assigned_gt, gt_labels, device,
            )
            total_obj_loss = total_obj_loss + obj_loss
            total_bbox_loss = total_bbox_loss + bbox_loss
            total_class_loss = total_class_loss + class_loss
            total_positives += num_pos

            if aux_preds is not None:
                aux_obj_loss, aux_bbox_loss, aux_class_loss, _ = self._image_losses(
                    aux_obj, aux_bbox_decoded, aux_class,
                    labels, bbox_targets, assigned_gt, gt_labels, device,
                )
                total_aux_loss = total_aux_loss + self._combine(
                    aux_obj_loss, aux_bbox_loss, aux_class_loss, multi_class,
                )

        # Normalize
        total_obj_loss = total_obj_loss / batch_size
        total_bbox_loss = total_bbox_loss / max(batch_size, 1)
        total_class_loss = total_class_loss / max(batch_size, 1)
        total_aux_loss = total_aux_loss / max(batch_size, 1)

        total_loss = self._combine(total_obj_loss, total_bbox_loss, total_class_loss, multi_class)
        if aux_preds is not None:
            total_loss = total_loss + self.aux_loss_weight * total_aux_loss

        return {
            "total_loss": total_loss,
            "objectness_loss": total_obj_loss.detach(),
            "bbox_loss": total_bbox_loss.detach(),
            "class_loss": total_class_loss.detach(),
            "aux_loss": total_aux_loss.detach(),
            "centerness_loss": torch.tensor(0.0),  # v2: removed, kept for compat
            "num_positives": total_positives,
        }
