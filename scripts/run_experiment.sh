#!/bin/bash
# Run one budgeted experiment end-to-end: train -> export -> eval mAP -> CPU bench.
#
# Usage: scripts/run_experiment.sh <TAG> [extra train args...]
#
# The literal line "RESULTS" is the completion marker the remote poller greps
# for, so it is printed only at the very end, after every step has succeeded.
set -euo pipefail

# Env-overridable defaults
DATASET=${DATASET:-hcap}
PRESET=${PRESET:-small}
BUDGET_MIN=${BUDGET_MIN:-15}
INPUT_SIZE=${INPUT_SIZE:-320}
EPOCHS=${EPOCHS:-10000}   # effectively unlimited; the minute budget stops first
# Invoke the venv python directly: on gpu3 the venv deliberately diverges from
# uv.lock (CUDA torch cu124 replaces the locked CPU build), and `uv run` would
# re-sync the env back to the broken lock on every invocation.
PY=${PY:-.venv/bin/python}

if [ $# -lt 1 ]; then
  echo "usage: $0 <TAG> [extra train args...]" >&2
  exit 1
fi
TAG=$1
shift

cd "$(dirname "$0")/.."   # repo root
RUN_DIR=runs/$TAG
mkdir -p "$RUN_DIR"

echo "=== [1/4] train ($PRESET on $DATASET, budget ${BUDGET_MIN}min) ==="
$PY -m boxvision.cli train \
  --preset "$PRESET" \
  --dataset "$DATASET" \
  --epochs "$EPOCHS" \
  --input-size "$INPUT_SIZE" \
  --max-minutes "$BUDGET_MIN" \
  --save-dir "runs/$TAG" \
  --device cuda \
  "$@" 2>&1 | tee "$RUN_DIR/train.log"

TRAIN_MIN=$(awk '$1 == "TRAIN_MINUTES:" {print $2}' "$RUN_DIR/train.log" | tail -n 1)
TRAIN_MIN=${TRAIN_MIN:-0.0}

# 2-way regime: trainings may overlap, but the measurement tail must run on a
# quiet machine — serialize export/eval/bench across all slots via a global lock.
exec 9>/tmp/boxvision-bench.lock
echo "waiting for bench lock..."
flock 9
echo "bench lock acquired"

echo "=== [2/4] export ==="
CKPT=$RUN_DIR/best.pt
if [ ! -f "$CKPT" ]; then
  CKPT=$RUN_DIR/last.pt
fi
if [ ! -f "$CKPT" ]; then
  echo "no checkpoint in $RUN_DIR (neither best.pt nor last.pt) -- training crashed" >&2
  exit 1
fi
$PY -m boxvision.cli export \
  --checkpoint "$CKPT" \
  --output "$RUN_DIR/model.onnx"

echo "=== [3/4] eval mAP ==="
$PY scripts/eval_onnx.py \
  --onnx "$RUN_DIR/model.onnx" \
  --dataset "$DATASET" \
  --input-size "$INPUT_SIZE" 2>&1 | tee "$RUN_DIR/eval.log"

# eval_onnx.py prints "  mAP@0.5:        79.07%" / "  mAP@0.5:0.95:   34.60%"
# (already percentages); exact first-field match keeps the two lines apart.
MAP50=$(awk '$1 == "mAP@0.5:" {gsub(/%/, "", $2); print $2}' "$RUN_DIR/eval.log" | tail -n 1)
MAP5095=$(awk '$1 == "mAP@0.5:0.95:" {gsub(/%/, "", $2); print $2}' "$RUN_DIR/eval.log" | tail -n 1)
MAP50=${MAP50:-0.00}
MAP5095=${MAP5095:-0.00}

echo "=== [4/4] CPU bench ==="
$PY scripts/bench_cpu.py \
  --onnx "$RUN_DIR/model.onnx" \
  --input-size "$INPUT_SIZE" 2>&1 | tee "$RUN_DIR/bench.log"

LAT=$(awk '$1 == "latency_ms_1t:" {print $2}' "$RUN_DIR/bench.log" | tail -n 1)
IPS=$(awk '$1 == "throughput_ips_8w:" {print $2}' "$RUN_DIR/bench.log" | tail -n 1)
PARAMS=$(awk '$1 == "params_k:" {print $2}' "$RUN_DIR/bench.log" | tail -n 1)

echo "RESULTS"
echo "mAP50:             $MAP50"
echo "mAP5095:           $MAP5095"
echo "latency_ms_1t:     $LAT"
echo "throughput_ips_8w: $IPS"
echo "params_k:          $PARAMS"
echo "train_minutes:     $TRAIN_MIN"
