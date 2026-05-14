"""
Evaluation metrics for BoxVision.

Computes:
- mAP@0.5 (mean Average Precision at IoU=0.5)
- Precision and Recall
- Per-image detection counts

Uses the standard PASCAL VOC evaluation protocol adapted for
class-agnostic detection (single class = "object").
"""

import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


def compute_iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between all pairs of boxes.

    Args:
        boxes_a: [N, 4] in (x1, y1, x2, y2)
        boxes_b: [M, 4] in (x1, y1, x2, y2)

    Returns:
        iou_matrix: [N, M]
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    # Expand for broadcasting
    a = boxes_a[:, None, :].expand(N, M, 4)
    b = boxes_b[None, :, :].expand(N, M, 4)

    # Intersection
    inter_x1 = torch.max(a[:, :, 0], b[:, :, 0])
    inter_y1 = torch.max(a[:, :, 1], b[:, :, 1])
    inter_x2 = torch.min(a[:, :, 2], b[:, :, 2])
    inter_y2 = torch.min(a[:, :, 3], b[:, :, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    area_a = (a[:, :, 2] - a[:, :, 0]) * (a[:, :, 3] - a[:, :, 1])
    area_b = (b[:, :, 2] - b[:, :, 0]) * (b[:, :, 3] - b[:, :, 1])
    union_area = area_a + area_b - inter_area

    return inter_area / (union_area + 1e-7)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using the 11-point interpolation method.

    Args:
        recalls: Sorted recall values
        precisions: Corresponding precision values

    Returns:
        AP value (0 to 1)
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_detections(
    all_pred_boxes: List[torch.Tensor],
    all_pred_scores: List[torch.Tensor],
    all_gt_boxes: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Evaluate detection results against ground truth.

    Args:
        all_pred_boxes: List of [Ki, 4] predicted boxes per image
        all_pred_scores: List of [Ki] predicted scores per image
        all_gt_boxes: List of [Mi, 4] ground truth boxes per image
        iou_threshold: IoU threshold for matching

    Returns:
        Dict with mAP50, precision, recall, num_predictions, num_gt
    """
    # Collect all predictions with image indices
    all_preds = []
    total_gt = 0

    for img_idx, (pred_boxes, pred_scores, gt_boxes) in enumerate(
        zip(all_pred_boxes, all_pred_scores, all_gt_boxes)
    ):
        for j in range(pred_boxes.shape[0]):
            all_preds.append({
                "img_idx": img_idx,
                "score": pred_scores[j].item(),
                "box": pred_boxes[j],
            })
        total_gt += gt_boxes.shape[0]

    if total_gt == 0:
        return {
            "mAP50": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_predictions": len(all_preds),
            "num_gt": 0,
        }

    # Sort by score (descending)
    all_preds.sort(key=lambda x: x["score"], reverse=True)

    # Track which GT boxes have been matched per image
    gt_matched = {}
    for img_idx, gt_boxes in enumerate(all_gt_boxes):
        gt_matched[img_idx] = [False] * gt_boxes.shape[0]

    # Compute TP/FP for each prediction
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for pred_idx, pred in enumerate(all_preds):
        img_idx = pred["img_idx"]
        pred_box = pred["box"].unsqueeze(0)
        gt_boxes = all_gt_boxes[img_idx]

        if gt_boxes.shape[0] == 0:
            fp[pred_idx] = 1
            continue

        # Compute IoU with all GT boxes in this image
        ious = compute_iou_matrix(pred_box, gt_boxes)[0]  # [M]

        # Find best matching GT
        best_iou, best_gt_idx = ious.max(dim=0)

        if best_iou >= iou_threshold and not gt_matched[img_idx][best_gt_idx]:
            tp[pred_idx] = 1
            gt_matched[img_idx][best_gt_idx] = True
        else:
            fp[pred_idx] = 1

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

    # Compute AP
    ap = compute_ap(recalls, precisions)

    # Overall precision and recall
    total_tp = tp.sum()
    total_fp = fp.sum()

    return {
        "mAP50": float(ap),
        "precision": float(total_tp / (total_tp + total_fp + 1e-7)),
        "recall": float(total_tp / (total_gt + 1e-7)),
        "num_predictions": len(all_preds),
        "num_gt": total_gt,
    }


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device: torch.device = None,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Run evaluation on a full dataset.

    Args:
        model: BoxVision model (will be set to eval mode)
        dataloader: Validation DataLoader
        device: Target device
        iou_threshold: IoU threshold for matching

    Returns:
        Evaluation metrics dict
    """
    device = device or torch.device("cpu")
    model.eval()

    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["images"].to(device)
        gt_boxes_batch = batch["boxes"]

        # predict() = forward + decode + NMS. forward() alone returns raw head tensors (ONNX-safe).
        results = model.predict(images)

        for i, result in enumerate(results):
            all_pred_boxes.append(result["boxes"].cpu())
            all_pred_scores.append(result["scores"].cpu())
            all_gt_boxes.append(gt_boxes_batch[i])

    return evaluate_detections(
        all_pred_boxes, all_pred_scores, all_gt_boxes,
        iou_threshold=iou_threshold,
    )
