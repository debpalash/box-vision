"""
Dataset loader for BoxVision.

Supports COCO-format annotations for class-agnostic box detection.
All object categories are collapsed into a single "object" class.

Includes real mosaic augmentation (4-image composite).

Expected directory structure:
    data/
    ├── images/
    │   ├── train/
    │   └── val/
    └── annotations/
        ├── train.json
        └── val.json
"""

import json
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BoxDataset(Dataset):
    """
    Bounding box dataset with optional class supervision and mosaic augmentation.

    Loads images and bounding box annotations in COCO format. When
    `class_agnostic=True` every annotation is collapsed to a single "object"
    class (matches the original class-agnostic BoxVision). When False, the
    real COCO category_id is remapped to a dense 0..C-1 index.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        input_size: Tuple[int, int] = (320, 320),
        transforms: Optional[Callable] = None,
        is_training: bool = True,
        mosaic: bool = False,
        class_agnostic: bool = True,
        mixup: bool = False,
        mixup_prob: float = 0.15,
        copy_paste: bool = False,
        copy_paste_prob: float = 0.3,
    ):
        """
        Args:
            image_dir: Path to image directory
            annotation_file: Path to COCO-format JSON annotation file
            input_size: Target (H, W) for resizing
            transforms: Albumentations transform pipeline
            is_training: Whether this is a training set
            mosaic: Enable mosaic augmentation (4-image composite)
            class_agnostic: If True, all categories collapse to label=0 (one class).
                If False, build a dense category mapping and emit per-box class indices.
        """
        self.image_dir = image_dir
        self.input_size = input_size
        self.is_training = is_training
        self.mosaic = mosaic and is_training  # Only during training
        self.class_agnostic = class_agnostic
        # Extra augmentations (training only). Applied AFTER the main mosaic/
        # standard transform pipeline, on the final normalized tensors.
        self.mixup = mixup and is_training
        self.mixup_prob = mixup_prob
        self.copy_paste = copy_paste and is_training
        self.copy_paste_prob = copy_paste_prob

        # Load COCO annotations
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        # Build image ID -> filename mapping
        self.images = {img["id"]: img for img in coco_data["images"]}

        # Build dense category mapping: category_id -> 0..C-1, and the inverse names list.
        # Roboflow exports often include a "supercategory" entry (id 0) with no annotations;
        # filter to categories actually referenced and sort by id for determinism.
        raw_categories = sorted(coco_data.get("categories", []), key=lambda c: c["id"])
        referenced_ids = {ann["category_id"] for ann in coco_data.get("annotations", [])}
        kept = [c for c in raw_categories if c["id"] in referenced_ids]
        self.category_id_to_idx = {c["id"]: i for i, c in enumerate(kept)}
        self.class_names = [c["name"] for c in kept]
        self.num_classes = 1 if class_agnostic else len(self.class_names)

        # Build image ID -> list of (bbox, dense_label) annotations
        self.annotations = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = {"boxes": [], "labels": []}

            # COCO format: [x, y, w, h] -> convert to [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:  # Skip degenerate boxes
                self.annotations[img_id]["boxes"].append([x, y, x + w, y + h])
                if class_agnostic:
                    self.annotations[img_id]["labels"].append(0)
                else:
                    self.annotations[img_id]["labels"].append(
                        self.category_id_to_idx[ann["category_id"]]
                    )

        # Only keep images that exist on disk
        self.image_ids = []
        for img_id, img_info in self.images.items():
            img_path = os.path.join(self.image_dir, img_info["file_name"])
            if os.path.exists(img_path):
                self.image_ids.append(img_id)

        # Post-mosaic transforms (applied after mosaic compositing)
        if transforms is not None:
            self.transforms = transforms
        elif is_training:
            self.transforms = self._train_transforms()
        else:
            self.transforms = self._val_transforms()

    def _load_image_and_boxes(self, idx: int):
        """Load a single image with its boxes and class labels (before transforms)."""
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann = self.annotations.get(img_id, {"boxes": [], "labels": []})
        if ann["boxes"]:
            boxes = np.array(ann["boxes"], dtype=np.float32)
            labels = np.array(ann["labels"], dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        return image, boxes, labels, img_id

    def _mosaic_4(self, idx: int):
        """
        Mosaic augmentation: compose 4 images into a single training sample.

        Places 4 images around a random center point within the output canvas,
        then clips boxes to the visible region.

        Returns:
            image: [H, W, 3] composited RGB image (uint8)
            boxes: [N, 4] concatenated boxes in xyxy format
        """
        target_h, target_w = self.input_size

        # Random center point for the 4-image layout
        cx = int(random.uniform(target_w * 0.25, target_w * 0.75))
        cy = int(random.uniform(target_h * 0.25, target_h * 0.75))

        # Pick 3 random partners + the requested index
        indices = [idx] + random.choices(range(len(self)), k=3)

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        all_boxes = []
        all_labels = []

        # Quadrants: top-left, top-right, bottom-left, bottom-right
        placements = [
            # (canvas x1, y1, x2, y2) for each quadrant
            (0, 0, cx, cy),
            (cx, 0, target_w, cy),
            (0, cy, cx, target_h),
            (cx, cy, target_w, target_h),
        ]

        for i, (quad_idx, (qx1, qy1, qx2, qy2)) in enumerate(zip(indices, placements)):
            image, boxes, labels, _ = self._load_image_and_boxes(quad_idx)
            h, w = image.shape[:2]
            quad_w = qx2 - qx1
            quad_h = qy2 - qy1

            if quad_w <= 0 or quad_h <= 0:
                continue

            # Scale image to fit the quadrant
            scale = min(quad_w / w, quad_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w <= 0 or new_h <= 0:
                continue

            resized = cv2.resize(image, (new_w, new_h))

            # Paste into canvas
            paste_x = qx1
            paste_y = qy1
            paste_h = min(new_h, quad_h)
            paste_w = min(new_w, quad_w)
            canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = resized[:paste_h, :paste_w]

            # Transform boxes
            if len(boxes) > 0:
                scaled_boxes = boxes * scale
                scaled_boxes[:, 0] += paste_x
                scaled_boxes[:, 1] += paste_y
                scaled_boxes[:, 2] += paste_x
                scaled_boxes[:, 3] += paste_y

                # Clip to canvas
                scaled_boxes[:, 0] = np.clip(scaled_boxes[:, 0], 0, target_w)
                scaled_boxes[:, 1] = np.clip(scaled_boxes[:, 1], 0, target_h)
                scaled_boxes[:, 2] = np.clip(scaled_boxes[:, 2], 0, target_w)
                scaled_boxes[:, 3] = np.clip(scaled_boxes[:, 3], 0, target_h)

                # Filter degenerate boxes after clipping
                w_box = scaled_boxes[:, 2] - scaled_boxes[:, 0]
                h_box = scaled_boxes[:, 3] - scaled_boxes[:, 1]
                keep = (w_box > 2) & (h_box > 2)
                scaled_boxes = scaled_boxes[keep]
                scaled_labels = labels[keep]

                if len(scaled_boxes) > 0:
                    all_boxes.append(scaled_boxes)
                    all_labels.append(scaled_labels)

        if all_boxes:
            all_boxes = np.concatenate(all_boxes, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_labels = np.zeros((0,), dtype=np.int64)

        return canvas, all_boxes, all_labels

    def _train_transforms(self) -> A.Compose:
        """Training augmentation pipeline."""
        return A.Compose(
            [
                A.LongestMaxSize(max_size=max(self.input_size)),
                A.PadIfNeeded(
                    min_height=self.input_size[0],
                    min_width=self.input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=114,
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",  # x1, y1, x2, y2
                min_area=1.0,
                min_visibility=0.3,
                label_fields=["labels"],
            ),
        )

    def _mosaic_transforms(self) -> A.Compose:
        """Post-mosaic transforms: just color jitter + normalize (no resize, already sized)."""
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=1.0,
                min_visibility=0.3,
                label_fields=["labels"],
            ),
        )

    def _val_transforms(self) -> A.Compose:
        """Validation/inference transforms (no augmentation)."""
        return A.Compose(
            [
                A.LongestMaxSize(max_size=max(self.input_size)),
                A.PadIfNeeded(
                    min_height=self.input_size[0],
                    min_width=self.input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=114,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=1.0,
                min_visibility=0.3,
                label_fields=["labels"],
            ),
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns dict with:
            image:    [3, H, W] normalized tensor
            boxes:    [N, 4] tensor in (x1, y1, x2, y2) format
            labels:   [N] long tensor of class indices (all zeros if class_agnostic)
            image_id: original COCO image ID
        """
        sample = self._load_one_sample(idx)

        if self.mixup and random.random() < self.mixup_prob:
            partner = self._load_one_sample(random.randint(0, len(self) - 1))
            sample = self._apply_mixup(sample, partner)

        if self.copy_paste and random.random() < self.copy_paste_prob:
            partner = self._load_one_sample(random.randint(0, len(self) - 1))
            sample = self._apply_copy_paste(sample, partner)

        return sample

    def _apply_mixup(self, s1: dict, s2: dict) -> dict:
        """Linear blend of two image tensors; concat boxes/labels.

        We sample α from a Beta distribution centered around 0.5 (matching the
        standard mixup recipe). Boxes carry over unmodified because both images
        have the same letterboxed dimensions.
        """
        alpha = float(np.random.beta(8.0, 8.0))  # tighter around 0.5
        s1["image"] = s1["image"] * alpha + s2["image"] * (1.0 - alpha)
        if s2["boxes"].numel() > 0:
            s1["boxes"] = torch.cat([s1["boxes"], s2["boxes"]], dim=0)
            s1["labels"] = torch.cat([s1["labels"], s2["labels"]], dim=0)
        return s1

    def _apply_copy_paste(self, s1: dict, s2: dict) -> dict:
        """Copy up to 3 box-cropped pixel regions from s2 into s1 in-place."""
        if s2["boxes"].numel() == 0:
            return s1
        n_objs = s2["boxes"].shape[0]
        n_take = min(3, n_objs)
        pick = torch.randperm(n_objs)[:n_take]
        new_boxes = s2["boxes"][pick]
        new_labels = s2["labels"][pick]
        H, W = s1["image"].shape[1:]
        added_boxes = []
        added_labels = []
        for i, box in enumerate(new_boxes):
            x1, y1, x2, y2 = box.tolist()
            xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
            xi1, yi1 = max(0, xi1), max(0, yi1)
            xi2, yi2 = min(W, xi2), min(H, yi2)
            if xi2 - xi1 < 2 or yi2 - yi1 < 2:
                continue
            s1["image"][:, yi1:yi2, xi1:xi2] = s2["image"][:, yi1:yi2, xi1:xi2]
            added_boxes.append(torch.tensor([xi1, yi1, xi2, yi2], dtype=torch.float32))
            added_labels.append(new_labels[i])
        if added_boxes:
            s1["boxes"] = torch.cat([s1["boxes"], torch.stack(added_boxes)], dim=0)
            s1["labels"] = torch.cat([s1["labels"], torch.stack(added_labels)], dim=0)
        return s1

    def _load_one_sample(self, idx: int) -> dict:
        """Produce one normalized sample via the mosaic/standard pipeline."""
        if self.mosaic and random.random() < 0.8:
            # Mosaic: 80% probability during training
            image, boxes, labels = self._mosaic_4(idx)

            mosaic_t = self._mosaic_transforms()
            transformed = mosaic_t(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else [],
            )
            img_id = self.image_ids[idx]
        else:
            # Standard single-image path
            image, boxes, labels, img_id = self._load_image_and_boxes(idx)

            transformed = self.transforms(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else [],
            )

        image = transformed["image"]  # [3, H, W] tensor

        if len(transformed["bboxes"]) > 0:
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
        }

    def set_mosaic(self, enabled: bool):
        """Toggle mosaic on/off (for late-epoch disabling)."""
        self.mosaic = enabled and self.is_training

    def set_input_size(self, input_size: Tuple[int, int]):
        """Resize the augmentation pipeline to a new (H, W). Used for multi-scale training."""
        self.input_size = input_size
        # Rebuild only the transforms that depend on input size
        if self.is_training:
            self.transforms = self._train_transforms()
        else:
            self.transforms = self._val_transforms()


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for variable-length box annotations.

    Stacks images into a batch tensor but keeps boxes and labels as lists
    (since each image can have a different number of objects).
    """
    images = torch.stack([item["image"] for item in batch])
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "image_ids": image_ids,
    }


def _worker_init_fn(worker_id: int):
    """Seed python/numpy RNGs per dataloader worker (torch already seeds its own
    per-worker generator); mosaic/mixup sampling uses `random`, so without this
    workers stay unseeded even when training is."""
    base = torch.initial_seed() % 2**31
    random.seed(base + worker_id)
    np.random.seed((base + worker_id) % 2**32)


def build_dataloader(
    image_dir: str,
    annotation_file: str,
    input_size: Tuple[int, int] = (320, 320),
    batch_size: int = 16,
    num_workers: int = 4,
    is_training: bool = True,
    transforms: Optional[Callable] = None,
    mosaic: bool = False,
    class_agnostic: bool = True,
    mixup: bool = False,
    copy_paste: bool = False,
) -> DataLoader:
    """Factory function to create a DataLoader."""
    dataset = BoxDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        input_size=input_size,
        transforms=transforms,
        is_training=is_training,
        mosaic=mosaic,
        class_agnostic=class_agnostic,
        mixup=mixup,
        copy_paste=copy_paste,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=is_training,
        worker_init_fn=_worker_init_fn,
    )
