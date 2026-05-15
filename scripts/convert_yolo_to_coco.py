"""
Convert a Label-Studio-style YOLO export into Roboflow-style COCO layout.

Input  (--src):
    src/
        classes.txt
        notes.json (optional)
        images/  *.jpeg|*.jpg|*.png
        labels/  *.txt   — YOLO format: class_id cx cy w h (normalized)

Output (--out):
    out/
        train/
            _annotations.coco.json
            *.jpg
        valid/
            _annotations.coco.json
            *.jpg

By default all classes are collapsed to a single "object" class (matches our
class-agnostic detector). Pass --keep-classes to preserve original class ids.

Split is deterministic via --seed.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to extracted YOLO export dir")
    ap.add_argument("--out", required=True, help="Output dir (will contain train/, valid/)")
    ap.add_argument("--val-fraction", type=float, default=0.2, help="Fraction reserved for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-classes", action="store_true",
                    help="Preserve original classes instead of collapsing to 1 'object' class")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    classes_file = src / "classes.txt"
    images_dir = src / "images"
    labels_dir = src / "labels"

    if not classes_file.exists():
        print(f"error: {classes_file} not found", file=sys.stderr)
        return 1
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"error: expected {images_dir} and {labels_dir} subdirs", file=sys.stderr)
        return 1

    class_names = [l.strip() for l in classes_file.read_text().splitlines() if l.strip()]
    if args.keep_classes:
        categories = [{"id": i + 1, "name": n, "supercategory": "object"}
                      for i, n in enumerate(class_names)]
    else:
        categories = [{"id": 1, "name": "object", "supercategory": "object"}]

    # Match images to label files by stem; skip images with no label or empty label.
    image_paths = sorted([p for p in images_dir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    pairs = []
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        pairs.append((img_path, label_path))

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = int(len(pairs) * args.val_fraction)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"total paired: {len(pairs)}  train: {len(train_pairs)}  val: {len(val_pairs)}")

    # Write splits.
    for split_name, split_pairs in [("train", train_pairs), ("valid", val_pairs)]:
        split_dir = out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_split(split_dir, split_pairs, categories, args.keep_classes, class_names)

    print(f"\nDone. Layout:")
    print(f"  {out}/train/  {len(train_pairs)} images")
    print(f"  {out}/valid/  {len(val_pairs)} images")
    print(f"  categories: {len(categories)} ({'preserved' if args.keep_classes else 'collapsed to object'})")
    return 0


def _write_split(split_dir, pairs, categories, keep_classes, class_names):
    images, annotations = [], []
    ann_id = 1
    for img_id, (img_path, label_path) in enumerate(pairs, start=1):
        # Copy the image; use a deterministic flat name keyed by stem.
        dst_name = img_path.name
        shutil.copy2(img_path, split_dir / dst_name)

        with Image.open(img_path) as im:
            W, H = im.size

        images.append({
            "id": img_id,
            "file_name": dst_name,
            "height": H,
            "width": W,
        })

        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_idx = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                continue
            # Convert + clamp to image bounds (YOLO labels can have slight
            # overflow due to annotation precision).
            x1 = max(0.0, (cx - w / 2) * W)
            y1 = max(0.0, (cy - h / 2) * H)
            x2 = min(float(W), (cx + w / 2) * W)
            y2 = min(float(H), (cy + h / 2) * H)
            x_abs, y_abs = x1, y1
            w_abs, h_abs = x2 - x1, y2 - y1
            if w_abs <= 1 or h_abs <= 1:
                continue
            cat_id = (cls_idx + 1) if keep_classes else 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x_abs, y_abs, w_abs, h_abs],
                "area": w_abs * h_abs,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": "Converted from YOLO export"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))


if __name__ == "__main__":
    sys.exit(main())
