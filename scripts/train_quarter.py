"""
Train on a fraction of road-signs with the known-working tiny config until best mAP.

Train: SUBSET_FRACTION of 1377 images (deterministic seed)
Val:   full 488 images
Epochs: 100, save best on raw mAP@0.5.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from torch.utils.data import Subset, DataLoader

from boxvision.config import tiny_config, TrainConfig
from boxvision.dataset import collate_fn
from boxvision.train import Trainer


class MosaicAwareSubset(Subset):
    """Subset that forwards set_mosaic() to the underlying dataset."""

    def set_mosaic(self, enabled: bool) -> None:
        if hasattr(self.dataset, "set_mosaic"):
            self.dataset.set_mosaic(enabled)


SUBSET_FRACTION = 0.50
SEED = 42


def main() -> None:
    model_config = tiny_config(pretrained_backbone=True)
    train_config = TrainConfig(
        dataset="road-signs",
        epochs=100,
        eval_interval=5,
        save_interval=10,
        save_dir="./runs/half-100ep",
        loss_objectness_weight=1.0,
        ema_decay=0.9999,
        ema_warmup_steps=0,
    )

    trainer = Trainer(model_config, train_config)

    # Build standard dataloaders, then replace the train loader with a 25% subset.
    trainer.build_dataloaders()

    full_train = trainer.train_loader.dataset
    n = len(full_train)
    k = max(1, int(n * SUBSET_FRACTION))
    rng = random.Random(SEED)
    indices = rng.sample(range(n), k)
    subset = MosaicAwareSubset(full_train, indices)

    print(f"\nSubsampled train: {k}/{n} ({100*SUBSET_FRACTION:.0f}%)  seed={SEED}")
    print(f"Val (unchanged):  {len(trainer.val_loader.dataset)}")

    trainer.train_loader = DataLoader(
        subset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Run training (build_dataloaders runs again inside .train() — patch that out by
    # subclassing or just override the method on the instance for this run).
    trainer.build_dataloaders = lambda: None  # type: ignore[assignment]
    trainer.train()


if __name__ == "__main__":
    main()
