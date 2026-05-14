"""
Autonomous 4-strategy sweep on road-signs (full data).

Runs four training configurations sequentially, writes a JSON summary at the
end. Each strategy saves to its own runs/ subdir; best.pt is selected by
max(raw, ema) mAP@0.5 inside Trainer.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxvision.config import tiny_config, small_config, TrainConfig
from boxvision.train import Trainer


STRATEGIES = [
    {
        "name": "S1-tiny-50ep-baseline",
        "save_dir": "./runs/sweep/S1-tiny-50ep-baseline",
        "model_factory": lambda: tiny_config(pretrained_backbone=True),
        "train_overrides": dict(
            epochs=50,
            loss_objectness_weight=1.0,
            ema_decay=0.9999,
            ema_warmup_steps=0,
            tal_use_soft_labels=True,
        ),
    },
    {
        "name": "S2-tiny-30ep-lr0.02",
        "save_dir": "./runs/sweep/S2-tiny-30ep-lr0.02",
        "model_factory": lambda: tiny_config(pretrained_backbone=True),
        "train_overrides": dict(
            epochs=30,
            learning_rate=0.02,
            loss_objectness_weight=1.0,
            ema_decay=0.9999,
            ema_warmup_steps=0,
            tal_use_soft_labels=True,
        ),
    },
    {
        "name": "S3-tiny-30ep-hard-labels",
        "save_dir": "./runs/sweep/S3-tiny-30ep-hard-labels",
        "model_factory": lambda: tiny_config(pretrained_backbone=True),
        "train_overrides": dict(
            epochs=30,
            loss_objectness_weight=1.0,
            ema_decay=0.9999,
            ema_warmup_steps=0,
            tal_use_soft_labels=False,
        ),
    },
    {
        "name": "S4-small-30ep-baseline",
        "save_dir": "./runs/sweep/S4-small-30ep-baseline",
        "model_factory": lambda: small_config(pretrained_backbone=True),
        "train_overrides": dict(
            epochs=30,
            loss_objectness_weight=1.0,
            ema_decay=0.9999,
            ema_warmup_steps=0,
            tal_use_soft_labels=True,
        ),
    },
]


def run_strategy(strategy: dict) -> dict:
    print(f"\n{'#'*70}")
    print(f"# {strategy['name']}")
    print(f"{'#'*70}\n")

    t0 = time.time()
    model_config = strategy["model_factory"]()
    train_config = TrainConfig(
        dataset="road-signs",
        eval_interval=5,
        save_interval=10,
        save_dir=strategy["save_dir"],
        **strategy["train_overrides"],
    )
    trainer = Trainer(model_config, train_config)
    trainer.train()

    elapsed = time.time() - t0
    result = {
        "name": strategy["name"],
        "save_dir": strategy["save_dir"],
        "elapsed_sec": round(elapsed, 1),
        "best_mAP50": trainer.best_metric,
    }
    print(f"\n>>> {strategy['name']} done in {elapsed/60:.1f} min, "
          f"best mAP@0.5={trainer.best_metric:.4f}")
    return result


def main() -> None:
    results = []
    overall_t0 = time.time()

    for strategy in STRATEGIES:
        try:
            results.append(run_strategy(strategy))
        except Exception as exc:
            print(f"\n!!! {strategy['name']} FAILED: {exc.__class__.__name__}: {exc}")
            results.append({
                "name": strategy["name"],
                "error": f"{exc.__class__.__name__}: {exc}",
            })
        # Persist incrementally so we always have something to read
        out = {
            "elapsed_total_sec": round(time.time() - overall_t0, 1),
            "results": results,
        }
        with open("/tmp/boxvision-sweep-results.json", "w") as f:
            json.dump(out, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE in {(time.time() - overall_t0)/60:.1f} min")
    print(f"{'='*70}")
    for r in results:
        if "best_mAP50" in r:
            print(f"  {r['name']:35s}  mAP@0.5={r['best_mAP50']:.4f}  ({r['elapsed_sec']/60:.1f} min)")
        else:
            print(f"  {r['name']:35s}  ERROR: {r.get('error', '?')}")

    best = max(
        (r for r in results if "best_mAP50" in r),
        key=lambda r: r["best_mAP50"],
        default=None,
    )
    if best:
        print(f"\nWINNER: {best['name']}  best mAP@0.5={best['best_mAP50']:.4f}")
        print(f"        Checkpoint: {best['save_dir']}/best.pt")


if __name__ == "__main__":
    main()
