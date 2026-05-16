"""
Train a P2+aug teacher on a registered dataset (mirrors the G recipe).
Used as the source for offline KD pseudo-labeling.

Usage:
    python scripts/train_teacher.py --dataset road-signs --tag rs-teacher-416-200ep
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxvision.config import small_config, TrainConfig
from boxvision.train import Trainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--input-size", type=int, default=416)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--mosaic-off-epochs", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    args = p.parse_args()

    model_config = small_config(
        pretrained_backbone=True,
        use_p2=True,
        strides=[4, 8, 16, 32],
        input_size=(args.input_size, args.input_size),
    )

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
        mixup=True,
        copy_paste=True,
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
