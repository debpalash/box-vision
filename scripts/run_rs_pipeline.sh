#!/bin/bash
# Road-signs KD pipeline: train teacher → distill pseudo-labels → train tiny+P2 student.
# All three phases write to runs/rs-pipeline.log so a single boxship wait can
# block on PIPELINE_DONE.
set -euo pipefail

cd ~/box-vision
log=runs/rs-pipeline.log

{
  echo "=== PHASE 1: teacher (small+P2+aug @ 416, 200ep) ==="
  .venv/bin/python scripts/train_teacher.py \
    --dataset road-signs \
    --tag rs-teacher-416-200ep \
    --epochs 200 \
    --input-size 416 \
    --batch-size 64 \
    --eval-interval 20 \
    --mosaic-off-epochs 20 \
    --device cuda

  echo "=== PHASE 2: distill pseudo-labels ==="
  .venv/bin/python scripts/distill_pseudo_labels.py \
    --teacher runs/rs-teacher-416-200ep/best.pt \
    --dataset road-signs \
    --out datasets/road-signs-distilled \
    --conf 0.40 \
    --iou-with-gt 0.5 \
    --max-per-image 20

  echo "=== PHASE 3: student (tiny+P2+light-aug @ 320, 200ep) ==="
  .venv/bin/python scripts/train_kd_student.py \
    --tag rs-tiny-kd-200ep \
    --dataset road-signs-distilled \
    --epochs 200 \
    --batch-size 64 \
    --eval-interval 20 \
    --mosaic-off-epochs 40 \
    --device cuda \
    --use-p2 \
    --light-aug

  echo PIPELINE_DONE
} >> "$log" 2>&1
