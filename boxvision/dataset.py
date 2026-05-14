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
    Class-agnostic bounding box dataset with optional mosaic augmentation.

    Loads images and bounding box annotations in COCO format.
    All categories are treated as a single "object" class — we only
    care about locating boxes, not classifying them.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        input_size: Tuple[int, int] = (320, 320),
        transforms: Optional[Callable] = None,
        is_training: bool = True,
        mosaic: bool = False,
    ):
        """
        Args:
            image_dir: Path to image directory
            annotation_file: Path to COCO-format JSON annotation file
            input_size: Target (H, W) for resizing
            transforms: Albumentations transform pipeline
            is_training: Whether this is a training set
            mosaic: Enable mosaic augmentation (4-image composite)
        """
        self.image_dir = image_dir
        self.input_size = input_size
        self.is_training = is_training
        self.mosaic = mosaic and is_training  # Only during training

        # Load COCO annotations
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        # Build image ID -> filename mapping
        self.images = {img["id"]: img for img in coco_data["images"]}

        # Build image ID -> list of bbox annotations
        self.annotations = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []

            # COCO format: [x, y, w, h] -> convert to [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:  # Skip degenerate boxes
                self.annotations[img_id].append([x, y, x + w, y + h])

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
        """Load a single image and its boxes (before any transforms)."""
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = self.annotations.get(img_id, [])
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

        return image, boxes, img_id

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

        # Quadrants: top-left, top-right, bottom-left, bottom-right
        placements = [
            # (canvas x1, y1, x2, y2) for each quadrant
            (0, 0, cx, cy),
            (cx, 0, target_w, cy),
            (0, cy, cx, target_h),
            (cx, cy, target_w, target_h),
        ]

        for i, (quad_idx, (qx1, qy1, qx2, qy2)) in enumerate(zip(indices, placements)):
            image, boxes, _ = self._load_image_and_boxes(quad_idx)
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

                if len(scaled_boxes) > 0:
                    all_boxes.append(scaled_boxes)

        if all_boxes:
            all_boxes = np.concatenate(all_boxes, axis=0)
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)

        return canvas, all_boxes

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
        Returns:
            dict with:
                'image': [3, H, W] normalized tensor
                'boxes': [N, 4] tensor in (x1, y1, x2, y2) format
                'image_id': original image ID
        """
        if self.mosaic and random.random() < 0.8:
            # Mosaic: 80% probability during training
            image, boxes = self._mosaic_4(idx)
            labels = np.zeros(len(boxes), dtype=np.int64)

            mosaic_t = self._mosaic_transforms()
            transformed = mosaic_t(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else [],
            )
            img_id = self.image_ids[idx]
        else:
            # Standard single-image path
            image, boxes, img_id = self._load_image_and_boxes(idx)
            labels = np.zeros(len(boxes), dtype=np.int64)

            transformed = self.transforms(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else [],
            )

        image = transformed["image"]  # [3, H, W] tensor

        if len(transformed["bboxes"]) > 0:
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "image": image,
            "boxes": boxes,
            "image_id": img_id,
        }

    def set_mosaic(self, enabled: bool):
        """Toggle mosaic on/off (for late-epoch disabling)."""
        self.mosaic = enabled and self.is_training


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for variable-length box annotations.

    Stacks images into a batch tensor but keeps boxes as a list
    (since each image can have different number of boxes).
    """
    images = torch.stack([item["image"] for item in batch])
    boxes = [item["boxes"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    return {
        "images": images,
        "boxes": boxes,
        "image_ids": image_ids,
    }


def build_dataloader(
    image_dir: str,
    annotation_file: str,
    input_size: Tuple[int, int] = (320, 320),
    batch_size: int = 16,
    num_workers: int = 4,
    is_training: bool = True,
    transforms: Optional[Callable] = None,
    mosaic: bool = False,
) -> DataLoader:
    """Factory function to create a DataLoader."""
    dataset = BoxDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        input_size=input_size,
        transforms=transforms,
        is_training=is_training,
        mosaic=mosaic,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=is_training,
    )
