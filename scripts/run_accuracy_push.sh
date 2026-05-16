#!/bin/bash
# Accuracy-push (parallel): 4 student trainings, 2 subshells running concurrently
# on the 4090. Each subshell handles 1 Tiny + 1 Small to balance GPU contention.
#
# Subshell A: shapes-tiny-416-ms → rs-small-kd-416
# Subshell B: rs-tiny-416-ms     → shapes-small-kd-416
#
# Pipeline marker "PIPELINE_DONE" emitted only after both subshells finish.
set -euo pipefail
cd ~/box-vision
log=runs/accpush-pipeline.log
mkdir -p runs

run() {
    local tag=$1; shift
    echo "[$(date +%H:%M:%S)] START $tag"
    .venv/bin/python scripts/train_kd_student.py --tag "$tag" "$@" \
        --device cuda --batch-size 32 --eval-interval 20
    echo "[$(date +%H:%M:%S)] DONE  $tag"
}

# Common student knobs
COMMON_TINY="--preset tiny  --use-p2 --light-aug --input-size 416 --multiscale --epochs 300 --mosaic-off-epochs 60"
COMMON_SMALL="--preset small --use-p2 --light-aug --input-size 416            --epochs 300 --mosaic-off-epochs 60"

{
    echo "=== START parallel pipeline (2 subshells) ==="

    (
        run shapes-tiny-416-ms-300ep  --dataset shapes-distilled       $COMMON_TINY
        run rs-small-kd-416-300ep     --dataset road-signs-distilled  $COMMON_SMALL
    ) > runs/accpush-A.log 2>&1 &
    PID_A=$!

    (
        run rs-tiny-416-ms-300ep      --dataset road-signs-distilled  $COMMON_TINY
        run shapes-small-kd-416-300ep --dataset shapes-distilled       $COMMON_SMALL
    ) > runs/accpush-B.log 2>&1 &
    PID_B=$!

    echo "Subshell A pid=$PID_A; Subshell B pid=$PID_B"

    wait $PID_A && echo "Subshell A finished" || echo "Subshell A FAILED"
    wait $PID_B && echo "Subshell B finished" || echo "Subshell B FAILED"

    echo PIPELINE_DONE
} >> "$log" 2>&1
