"""
Diagnostic: rerun the EXACT config that produced the working 32.6% mAP checkpoint.

If this reaches non-zero mAP in 10 epochs, the regression is in one of the
flags that diverged (Ghost-FPN, deeper head, obj_weight=4.0).
If this also stays at 0% mAP, the regression is elsewhere — code-level.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxvision.config import tiny_config, TrainConfig
from boxvision.train import Trainer


def main() -> None:
    model_config = tiny_config(pretrained_backbone=True)

    train_config = TrainConfig(
        dataset="road-signs",
        epochs=10,
        eval_interval=2,
        save_dir="./runs/repro-old",
        # Reverted to the OLD-run values (which reached 32.6% mAP):
        loss_objectness_weight=1.0,
        ema_decay=0.9999,
        ema_warmup_steps=0,  # Match the old (no-warmup) behavior
    )

    Trainer(model_config, train_config).train()


if __name__ == "__main__":
    main()
