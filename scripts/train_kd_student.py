"""
Train tiny KD student on the distilled dataset (real GT + G@416 teacher
pseudo-labels). This is "offline" distillation: the teacher's extra boxes are
baked into the dataset, so training is identical to a normal run — the student
just sees ~48% more positives per image.

Variants supported via flags:
    --use-p2     add stride-4 (P2) head — gives the small-object eye
    --light-aug  drop mixup/copy-paste; mosaic only (recommended for tiny)

Usage:
    python scripts/train_kd_student.py --tag tiny-kd-200ep --epochs 200
    python scripts/train_kd_student.py --tag tiny-p2-light-200ep --use-p2 --light-aug
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxvision.config import tiny_config, TrainConfig
from boxvision.train import Trainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="tiny-kd-200ep")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--dataset", default="shapes-distilled")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--mosaic-off-epochs", type=int, default=20)
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--use-p2", action="store_true",
                   help="Add stride-4 head (gives the small-object eye G has)")
    p.add_argument("--light-aug", action="store_true",
                   help="Mosaic only; drop mixup and copy-paste (kinder to tiny capacity)")
    args = p.parse_args()

    overrides = {}
    if args.use_p2:
        overrides["use_p2"] = True
        overrides["strides"] = [4, 8, 16, 32]
    model_config = tiny_config(pretrained_backbone=True, **overrides)

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
        loss_objectness_weight=4.0,
        loss_bbox_weight=2.0,
        use_ema=True,
        ema_decay=0.99,
        ema_warmup_steps=50,
        device=args.device,
        amp=args.amp,
    )

    trainer = Trainer(model_config, train_config)
    trainer.train()


if __name__ == "__main__":
    main()
