"""
Parameterized training run. Use this for ablations across presets and data fractions.

Usage:
    python scripts/train_run.py --preset tiny  --fraction 1.0 --epochs 100 --tag tiny-full-100ep
    python scripts/train_run.py --preset small --fraction 1.0 --epochs 100 --tag small-full-100ep
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import Subset, DataLoader

from boxvision.config import tiny_config, small_config, TrainConfig
from boxvision.dataset import collate_fn
from boxvision.train import Trainer


class MosaicAwareSubset(Subset):
    """Subset that forwards set_mosaic() to the underlying dataset."""

    def set_mosaic(self, enabled: bool) -> None:
        if hasattr(self.dataset, "set_mosaic"):
            self.dataset.set_mosaic(enabled)


PRESETS = {"tiny": tiny_config, "small": small_config}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS.keys(), required=True)
    p.add_argument("--fraction", type=float, default=1.0, help="train data fraction (0-1)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--tag", type=str, required=True, help="save_dir suffix (runs/{tag})")
    p.add_argument("--mosaic-off-epochs", type=int, default=0,
                   help="Epochs at end with mosaic disabled. Default 0 (always on).")
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model_config = PRESETS[args.preset](pretrained_backbone=True)
    train_config = TrainConfig(
        dataset="road-signs",
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        save_interval=10,
        save_dir=f"./runs/{args.tag}",
        loss_objectness_weight=1.0,
        ema_decay=0.9999,
        ema_warmup_steps=0,
        mosaic_off_epochs=args.mosaic_off_epochs,
    )

    trainer = Trainer(model_config, train_config)
    trainer.build_dataloaders()

    if args.fraction < 1.0:
        full = trainer.train_loader.dataset
        n = len(full)
        k = max(1, int(n * args.fraction))
        indices = random.Random(args.seed).sample(range(n), k)
        subset = MosaicAwareSubset(full, indices)
        print(f"Subsampled train: {k}/{n} ({100*args.fraction:.0f}%) seed={args.seed}")
        trainer.train_loader = DataLoader(
            subset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    # Trainer.train() re-runs build_dataloaders — bypass it so our subset survives.
    trainer.build_dataloaders = lambda: None  # type: ignore[assignment]
    trainer.train()


if __name__ == "__main__":
    main()
