"""
Train a KD student (tiny or small) on a distilled dataset.

Flags:
    --preset tiny|small  pick student size (default tiny → 164K params; small → 842K)
    --use-p2             add stride-4 head for small-object recall
    --light-aug          mosaic-only (skip mixup + copy-paste) — kinder to tiny capacity
    --input-size N       train at NxN (default 320)
    --multiscale         per-epoch sampling from --multiscale-sizes

Usage:
    python scripts/train_kd_student.py --tag tiny-kd-200ep --dataset shapes-distilled
    python scripts/train_kd_student.py --tag tiny-kd-416-ms --use-p2 --light-aug \\
        --input-size 416 --multiscale --epochs 300
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxvision.config import tiny_config, small_config, TrainConfig
from boxvision.train import Trainer


PRESETS = {"tiny": tiny_config, "small": small_config}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="tiny-kd-200ep")
    p.add_argument("--preset", choices=PRESETS.keys(), default="tiny")
    p.add_argument("--dataset", default="shapes-distilled")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--input-size", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--mosaic-off-epochs", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--use-p2", action="store_true")
    p.add_argument("--light-aug", action="store_true")
    p.add_argument("--multiscale", action="store_true")
    p.add_argument("--multiscale-sizes", type=int, nargs="+",
                   default=[320, 416, 512])
    args = p.parse_args()

    overrides = {"input_size": (args.input_size, args.input_size)}
    if args.use_p2:
        overrides["use_p2"] = True
        overrides["strides"] = [4, 8, 16, 32]
    model_config = PRESETS[args.preset](pretrained_backbone=True, **overrides)

    train_config = TrainConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        save_interval=20,
        save_dir=f"./runs/{args.tag}",
        augment=True,
        mosaic=True,
        mosaic_off_epochs=args.mosaic_off_epochs,
        mixup=(not args.light_aug),
        copy_paste=(not args.light_aug),
        multiscale=args.multiscale,
        multiscale_sizes=args.multiscale_sizes,
        loss_objectness_weight=4.0,
        loss_bbox_weight=2.0,
        use_ema=True,
        ema_decay=0.99,
        ema_warmup_steps=50,
        device=args.device,
        amp=args.amp,
    )

    Trainer(model_config, train_config).train()


if __name__ == "__main__":
    main()
