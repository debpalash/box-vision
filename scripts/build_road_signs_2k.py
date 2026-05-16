"""
Merge road-signs train + val + test (2093 imgs total) into a single dataset,
then split 90/10 into new train/valid. Image files are *symlinked* (no copy)
into datasets/road-signs-2k/{train,valid}/ so the working set is ~free.

Usage: python scripts/build_road_signs_2k.py
"""

import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxvision.registry import load_dataset


def main():
    spec = load_dataset("road-signs")
    splits = [("train", spec.train), ("val", spec.val), ("test", spec.test)]
    out_root = Path("datasets/road-signs-2k").resolve()
    out_train = out_root / "train"
    out_val = out_root / "valid"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    # Merge with fresh IDs so we don't collide.
    merged_images = []
    merged_anns = []
    categories = None
    next_img_id = 1
    next_ann_id = 1
    src_path_by_new_id: dict[int, Path] = {}

    for split_name, split in splits:
        with open(split.annotations) as f:
            coco = json.load(f)
        if categories is None:
            categories = coco["categories"]
        img_dir = Path(split.images)
        id_remap = {}
        for img in coco["images"]:
            new_id = next_img_id
            id_remap[img["id"]] = new_id
            merged_images.append({
                "id": new_id,
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            })
            src_path_by_new_id[new_id] = img_dir / img["file_name"]
            next_img_id += 1
        for ann in coco["annotations"]:
            merged_anns.append({
                "id": next_ann_id,
                "image_id": id_remap[ann["image_id"]],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann.get("area", float(ann["bbox"][2] * ann["bbox"][3])),
                "iscrowd": ann.get("iscrowd", 0),
            })
            next_ann_id += 1

    print(f"Merged: {len(merged_images)} images, {len(merged_anns)} annotations")

    # 90/10 split, fixed seed.
    rng = random.Random(42)
    img_ids = [im["id"] for im in merged_images]
    rng.shuffle(img_ids)
    n_val = max(1, int(len(img_ids) * 0.10))
    val_ids = set(img_ids[:n_val])
    train_ids = set(img_ids[n_val:])
    print(f"Split: train={len(train_ids)} val={len(val_ids)}")

    def write_split(out_dir: Path, ids: set):
        imgs = [im for im in merged_images if im["id"] in ids]
        anns = [a for a in merged_anns if a["image_id"] in ids]
        for im in imgs:
            src = src_path_by_new_id[im["id"]]
            dst = out_dir / im["file_name"]
            if not dst.exists():
                # Symlink to avoid duplicating disk.
                try:
                    dst.symlink_to(src.resolve())
                except FileExistsError:
                    pass
        ann_path = out_dir / "_annotations.coco.json"
        with open(ann_path, "w") as f:
            json.dump({"images": imgs, "annotations": anns, "categories": categories}, f)
        print(f"  wrote {ann_path.name}: {len(imgs)} imgs, {len(anns)} anns")

    write_split(out_train, train_ids)
    write_split(out_val, val_ids)
    print(f"Dataset root: {out_root}")


if __name__ == "__main__":
    main()
