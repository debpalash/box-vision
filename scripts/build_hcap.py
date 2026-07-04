"""
Build the hcap dataset from Label Studio YOLO export zips.

Input (--src): a directory containing project-*.zip Label Studio YOLO exports
(each with classes.txt, images/, labels/). Default: ../datasets/hcap relative
to the box-vision root (the boxmodel workspace staging dir).

Output (--out, default datasets/hcap): Roboflow-style COCO layout —
    train/_annotations.coco.json + images
    valid/_annotations.coco.json + images

Pipeline:
  1. Extract every zip into a temp dir.
  2. Dedupe images by content hash across (and within) projects; warn when
     duplicates carry different labels (first-seen wins, sorted order).
  3. Sanity-check labels: 5 fields, class 0, coords in [0,1] (clamped later).
  4. Split train/valid deterministically (--seed 42), stratified per source
     project so every capture batch appears in both splits.
  5. Convert YOLO cx/cy/w/h -> COCO xywh abs pixels, single "object" category.

Deterministic: same zips + same seed => identical dataset.

Usage: python scripts/build_hcap.py [--src DIR] [--out DIR] [--val-fraction 0.15]
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def extract_zips(src: Path, workdir: Path) -> list[tuple[str, Path]]:
    """Extract each project zip; return (project_tag, extracted_dir) pairs."""
    zips = sorted(src.glob("*.zip"))
    if not zips:
        sys.exit(f"error: no .zip files in {src}")
    projects = []
    for z in zips:
        # "project-11-at-2026-07-04-..." -> tag "p11"
        parts = z.stem.split("-")
        tag = "p" + parts[1] if len(parts) > 1 and parts[1].isdigit() else z.stem[:8]
        dest = workdir / tag
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        projects.append((tag, dest))
        print(f"extracted {z.name} -> {tag}/")
    return projects


def collect_pairs(projects: list[tuple[str, Path]]):
    """Gather (project_tag, image_path, label_path) with content-hash dedupe."""
    seen: dict[str, tuple[str, str]] = {}  # img_md5 -> (tag, name, label_md5)
    pairs = []
    n_dupes = n_label_conflicts = n_missing_label = 0
    for tag, root in projects:
        images_dir, labels_dir = root / "images", root / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            sys.exit(f"error: {tag}: expected images/ and labels/ in export")
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                n_missing_label += 1
                continue
            img_md5 = hashlib.md5(img_path.read_bytes()).hexdigest()
            lbl_md5 = hashlib.md5(label_path.read_bytes()).hexdigest()
            if img_md5 in seen:
                n_dupes += 1
                if seen[img_md5][1] != lbl_md5:
                    n_label_conflicts += 1
                    print(f"  WARN duplicate image with different labels: "
                          f"{tag}/{img_path.name} vs {seen[img_md5][0]} (kept first)")
                continue
            seen[img_md5] = (f"{tag}/{img_path.name}", lbl_md5)
            pairs.append((tag, img_path, label_path))
    print(f"collected {len(pairs)} unique pairs "
          f"(dupes dropped: {n_dupes}, label conflicts: {n_label_conflicts}, "
          f"missing labels: {n_missing_label})")
    return pairs


def parse_label(label_path: Path, W: int, H: int):
    """YOLO lines -> list of COCO xywh boxes (clamped). Returns (boxes, n_bad)."""
    boxes, n_bad = [], 0
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            n_bad += 1
            continue
        try:
            cls_idx = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            n_bad += 1
            continue
        if cls_idx != 0:
            n_bad += 1
            continue
        x1 = max(0.0, (cx - w / 2) * W)
        y1 = max(0.0, (cy - h / 2) * H)
        x2 = min(float(W), (cx + w / 2) * W)
        y2 = min(float(H), (cy + h / 2) * H)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            n_bad += 1
            continue
        boxes.append([x1, y1, bw, bh])
    return boxes, n_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--src", default=str(root.parent / "datasets" / "hcap"))
    ap.add_argument("--out", default=str(root / "datasets" / "hcap"))
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"error: {out} already exists — remove it to rebuild")

    with tempfile.TemporaryDirectory() as td:
        projects = extract_zips(Path(args.src), Path(td))
        pairs = collect_pairs(projects)

        # Stratified split: shuffle within each project, take val slice from each.
        rng = random.Random(args.seed)
        train_pairs, val_pairs = [], []
        for tag, _ in projects:
            proj = sorted((p for p in pairs if p[0] == tag), key=lambda p: p[1].name)
            rng.shuffle(proj)
            n_val = max(1, int(len(proj) * args.val_fraction))
            val_pairs += proj[:n_val]
            train_pairs += proj[n_val:]
            print(f"  {tag}: {len(proj) - n_val} train / {n_val} val")

        categories = [{"id": 1, "name": "object", "supercategory": "object"}]
        for split_name, split_pairs in [("train", train_pairs), ("valid", val_pairs)]:
            split_dir = out / split_name
            split_dir.mkdir(parents=True)
            images, annotations = [], []
            ann_id = 1
            total_bad = 0
            for img_id, (tag, img_path, label_path) in enumerate(
                    sorted(split_pairs, key=lambda p: (p[0], p[1].name)), start=1):
                dst_name = f"{tag}-{img_path.name}"
                shutil.copy2(img_path, split_dir / dst_name)
                with Image.open(img_path) as im:
                    W, H = im.size
                images.append({"id": img_id, "file_name": dst_name,
                               "height": H, "width": W})
                boxes, n_bad = parse_label(label_path, W, H)
                total_bad += n_bad
                for bb in boxes:
                    annotations.append({
                        "id": ann_id, "image_id": img_id, "category_id": 1,
                        "bbox": bb, "area": bb[2] * bb[3], "iscrowd": 0,
                    })
                    ann_id += 1
            coco = {"info": {"description": "hcap — Label Studio box exports merged"},
                    "licenses": [], "categories": categories,
                    "images": images, "annotations": annotations}
            (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))
            print(f"{split_name}: {len(images)} images, {len(annotations)} boxes"
                  f" (skipped bad label lines: {total_bad})")

    print(f"Dataset root: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
