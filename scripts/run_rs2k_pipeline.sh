#!/bin/bash
# Road-signs 2K pipeline: build merged dataset → train teacher → distill → train tiny+P2 student.
set -euo pipefail
cd ~/box-vision
log=runs/rs2k-pipeline.log

{
  echo "=== PHASE 0: build merged 2K dataset ==="
  .venv/bin/python scripts/build_road_signs_2k.py

  echo "=== PHASE 1: teacher (small+P2+aug @ 416, 200ep) on 2K ==="
  .venv/bin/python scripts/train_teacher.py \
    --dataset road-signs-2k \
    --tag rs2k-teacher-416-200ep \
    --epochs 200 \
    --input-size 416 \
    --batch-size 64 \
    --eval-interval 20 \
    --mosaic-off-epochs 20 \
    --device cuda

  echo "=== PHASE 2: distill pseudo-labels ==="
  .venv/bin/python scripts/distill_pseudo_labels.py \
    --teacher runs/rs2k-teacher-416-200ep/best.pt \
    --dataset road-signs-2k \
    --out datasets/road-signs-2k-distilled \
    --conf 0.40 \
    --iou-with-gt 0.5 \
    --max-per-image 20

  echo "=== PHASE 3: student (tiny+P2+light-aug @ 320, 200ep) on 2K-distilled ==="
  .venv/bin/python scripts/train_kd_student.py \
    --tag rs2k-tiny-kd-200ep \
    --dataset road-signs-2k-distilled \
    --epochs 200 \
    --batch-size 64 \
    --eval-interval 20 \
    --mosaic-off-epochs 40 \
    --device cuda \
    --use-p2 \
    --light-aug

  echo PIPELINE_DONE
} >> "$log" 2>&1
