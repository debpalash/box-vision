"""
Evaluation for BoxVision using pycocotools.

Computes:
- mAP@0.5
- mAP@0.5:0.95 (COCO primary metric)
- precision, recall (at the best F1 point)
- per-class AP@0.5 (multi-class only)

Works for both class-agnostic (num_classes=1) and multi-class detection.
"""

import contextlib
import io
import json
import tempfile
from typing import List, Optional

import torch
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device: Optional[torch.device] = None,
    iou_threshold: float = 0.5,  # kept for compat; pycocotools reports both 0.5 and 0.5:0.95
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Evaluate a BoxVision model on a validation set using COCO metrics.

    Args:
        model: BoxVision (will be put in eval mode)
        dataloader: validation DataLoader (must yield batches with images, boxes, labels)
        device: target device
        iou_threshold: kept for backward compat — reported as mAP50 alongside mAP@0.5:0.95
        class_names: optional list of class names for per-class reporting

    Returns:
        dict with mAP50, mAP, precision, recall, num_predictions, num_gt, and
        (multi-class only) per_class_ap50
    """
    device = device or torch.device("cpu")
    model.eval()

    # Detect class-agnostic vs multi-class from model config.
    num_classes = getattr(model.config, "num_classes", 1)
    multi_class = num_classes > 1

    # Accumulate predictions and GT in COCO format.
    coco_images: List[dict] = []
    coco_annotations: List[dict] = []
    coco_predictions: List[dict] = []
    ann_id = 1  # COCO requires non-zero IDs

    # pycocotools uses 1-indexed category IDs; build dense mapping.
    if multi_class:
        categories = [
            {"id": i + 1, "name": class_names[i] if class_names else f"class_{i}"}
            for i in range(num_classes)
        ]
    else:
        categories = [{"id": 1, "name": "object"}]

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["images"].to(device)
        gt_boxes_batch = batch["boxes"]
        gt_labels_batch = batch.get("labels") or [None] * len(gt_boxes_batch)
        image_ids = batch["image_ids"]
        H, W = images.shape[2], images.shape[3]

        results = model.predict(images)

        for i, result in enumerate(results):
            img_id = int(image_ids[i])
            coco_images.append({"id": img_id, "height": H, "width": W})

            gt_boxes = gt_boxes_batch[i]
            gt_labels = gt_labels_batch[i]
            for j in range(gt_boxes.shape[0]):
                x1, y1, x2, y2 = gt_boxes[j].tolist()
                cat_id = (int(gt_labels[j].item()) + 1) if (multi_class and gt_labels is not None) else 1
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                })
                ann_id += 1

            pred_boxes = result["boxes"].cpu()
            pred_scores = result["scores"].cpu()
            pred_labels = result["labels"].cpu()
            for j in range(pred_boxes.shape[0]):
                x1, y1, x2, y2 = pred_boxes[j].tolist()
                cat_id = (int(pred_labels[j].item()) + 1) if multi_class else 1
                coco_predictions.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(pred_scores[j].item()),
                })

    num_gt = len(coco_annotations)
    num_predictions = len(coco_predictions)

    if num_gt == 0:
        return {
            "mAP50": 0.0, "mAP": 0.0, "precision": 0.0, "recall": 0.0,
            "num_predictions": num_predictions, "num_gt": 0,
        }

    # Empty predictions ⇒ skip pycocotools (it errors on empty result files).
    if num_predictions == 0:
        return {
            "mAP50": 0.0, "mAP": 0.0, "precision": 0.0, "recall": 0.0,
            "num_predictions": 0, "num_gt": num_gt,
        }

    coco_gt_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_gt:
        json.dump(coco_gt_dict, f_gt)
        gt_path = f_gt.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_pr:
        json.dump(coco_predictions, f_pr)
        pr_path = f_pr.name

    # Silence the chatty pycocotools prints; we'll surface our own summary.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pr_path)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    stats = ev.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    mAP = float(stats[0]) if stats[0] >= 0 else 0.0
    mAP50 = float(stats[1]) if stats[1] >= 0 else 0.0
    recall_100 = float(stats[8]) if stats[8] >= 0 else 0.0

    # Best-F1 precision proxy: COCOeval doesn't give a single P value, so derive
    # an order-of-magnitude estimate from precision tensor at IoU=0.5.
    # ev.eval["precision"] shape: [TxRxKxAxM] (T thresholds, R recall, K classes,
    # A area, M maxDets). Take the mean precision across recall at IoU=0.5.
    precisions = ev.eval["precision"]  # T R K A M
    # IoU thresholds start at 0.5 with step 0.05 → index 0 is 0.5
    p_at_iou50 = precisions[0, :, :, 0, -1]  # [R, K]
    p_valid = p_at_iou50[p_at_iou50 > -1]
    precision = float(p_valid.mean()) if p_valid.size > 0 else 0.0

    result: dict = {
        "mAP50": mAP50,
        "mAP": mAP,
        "precision": precision,
        "recall": recall_100,
        "num_predictions": num_predictions,
        "num_gt": num_gt,
    }

    if multi_class:
        per_class_ap50: dict = {}
        for k, cat in enumerate(categories):
            cls_p = precisions[0, :, k, 0, -1]
            cls_valid = cls_p[cls_p > -1]
            per_class_ap50[cat["name"]] = float(cls_valid.mean()) if cls_valid.size > 0 else 0.0
        result["per_class_ap50"] = per_class_ap50

    return result
