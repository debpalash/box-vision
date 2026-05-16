"""
Finalize the accuracy-push pipeline:
  1. Pull all 4 best.pt from remote
  2. Export each to ONNX (using ckpt's saved model_config)
  3. Evaluate each on the right val set
  4. Print summary table

Galleries are generated separately by test_inference.py and test_inference_rs.py.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VARIANTS = [
    # (run_tag, dataset, eval_input_size)
    ("shapes-tiny-416-ms-300ep",   "shapes",      416),
    ("shapes-small-kd-416-300ep",  "shapes",      416),
    ("rs-tiny-416-ms-300ep",       "road-signs",  416),
    ("rs-small-kd-416-300ep",      "road-signs",  416),
]


def run(cmd: str, cwd: Path = ROOT) -> str:
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise SystemExit(f"exit {res.returncode}: {cmd}")
    return res.stdout


def main() -> None:
    print("=== 1) Pulling checkpoints ===")
    boxship = ROOT / "boxship" / "boxship"
    for tag, _, _ in VARIANTS:
        local = ROOT / "runs" / tag
        local.mkdir(parents=True, exist_ok=True)
        out = local / "best.pt"
        cmd = f'set -a; source ./.boxship.env; set +a; {shlex.quote(str(boxship))} run "cat ~/box-vision/runs/{tag}/best.pt" > {shlex.quote(str(out))} 2>/dev/null'
        subprocess.run(cmd, shell=True, cwd=ROOT, check=True)
        print(f"  pulled runs/{tag}/best.pt  ({out.stat().st_size/1024:.0f} KB)")

    print("\n=== 2) Exporting to ONNX ===")
    for tag, _, _ in VARIANTS:
        out = ROOT / "runs" / tag / f"boxvision-{tag}.onnx"
        run(f".venv/bin/python -m boxvision.cli export "
            f"--checkpoint runs/{tag}/best.pt --output {shlex.quote(str(out))} 2>&1 | tail -2")
        print(f"  runs/{tag}/{out.name}")

    print("\n=== 3) Evaluating ===")
    print(f"{'Variant':<32} {'Dataset':<14} {'mAP@0.5':>10} {'mAP@.5:.95':>12} {'avg_ms':>10}")
    print("-" * 84)
    results = []
    for tag, dataset, sz in VARIANTS:
        onnx = ROOT / "runs" / tag / f"boxvision-{tag}.onnx"
        out = run(f".venv/bin/python scripts/eval_onnx.py --onnx {shlex.quote(str(onnx))} "
                  f"--dataset {dataset} --input-size {sz} --conf 0.05 --nms-iou 0.50 2>&1 | tail -6")
        map5 = map95 = ms = "?"
        for line in out.splitlines():
            ls = line.strip()
            if ls.startswith("mAP@0.5:") and not ls.startswith("mAP@0.5:0.95"):
                map5 = ls.split()[-1]
            elif ls.startswith("mAP@0.5:0.95:"):
                map95 = ls.split()[-1]
            elif ls.startswith("avg latency:"):
                ms = ls.split()[-2] + "ms"
        results.append((tag, dataset, map5, map95, ms))
        print(f"{tag:<32} {dataset:<14} {map5:>10} {map95:>12} {ms:>10}")

    print("\nWrote ONNX models alongside best.pt in runs/<tag>/")


if __name__ == "__main__":
    main()
