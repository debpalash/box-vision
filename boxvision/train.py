"""
Training loop v2 for BoxVision.

v2 changes:
- EMA (exponential moving average) support
- Mosaic scheduling (off for last N epochs)
- TAL-compatible loss
- Simplified (no centerness loss term)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Optional

from .model import BoxVision, ModelEMA, build_model
from .losses import BoxVisionLoss
from .dataset import build_dataloader
from .config import ModelConfig, TrainConfig
from .evaluate import evaluate_model
from .registry import load_dataset, verify_dataset


class CosineWarmupScheduler:
    """Cosine annealing LR with linear warmup."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 warmup_lr_ratio: float = 0.001):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr_ratio = warmup_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            alpha = epoch / max(self.warmup_epochs, 1)
            factor = self.warmup_lr_ratio + (1 - self.warmup_lr_ratio) * alpha
        else:
            import math
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            factor = 0.5 * (1 + math.cos(math.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * factor


class Trainer:
    """BoxVision v2 training manager."""

    def __init__(
        self,
        model_config: ModelConfig = None,
        train_config: TrainConfig = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.train_config = train_config or TrainConfig()

        self.device = torch.device(self.train_config.device)
        self.model = build_model(self.model_config).to(self.device)

        # EMA
        self.ema = None
        if self.train_config.use_ema:
            self.ema = ModelEMA(
                self.model,
                decay=self.train_config.ema_decay,
                warmup_steps=self.train_config.ema_warmup_steps,
            )
            print(
                f"EMA enabled (decay={self.train_config.ema_decay}, "
                f"warmup_steps={self.train_config.ema_warmup_steps})"
            )

        # Loss
        self.criterion = BoxVisionLoss(
            focal_alpha=self.train_config.focal_alpha,
            focal_gamma=self.train_config.focal_gamma,
            objectness_weight=self.train_config.loss_objectness_weight,
            bbox_weight=self.train_config.loss_bbox_weight,
            strides=self.model_config.strides,
            tal_topk=self.train_config.tal_topk,
            tal_alpha=self.train_config.tal_alpha,
            tal_beta=self.train_config.tal_beta,
            use_soft_labels=self.train_config.tal_use_soft_labels,
        )

        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            momentum=self.train_config.momentum,
            weight_decay=self.train_config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.train_config.warmup_epochs,
            total_epochs=self.train_config.epochs,
            warmup_lr_ratio=self.train_config.warmup_lr_ratio,
        )

        # AMP
        self.scaler = GradScaler(enabled=self.train_config.amp)

        self.best_metric = 0.0
        self.start_epoch = 0

    def build_dataloaders(self):
        """
        Build training and validation dataloaders from the dataset registry.

        Reads the dataset spec from datasets.yaml via boxvision.registry.
        The dataset name is taken from TrainConfig.dataset.
        """
        spec = load_dataset(self.train_config.dataset)
        errors = verify_dataset(spec)
        if errors:
            raise FileNotFoundError(
                f"Dataset '{spec.name}' has missing paths:\n  - " + "\n  - ".join(errors)
            )

        if spec.format != "coco":
            raise NotImplementedError(
                f"Dataset '{spec.name}' uses format '{spec.format}'. "
                f"Only 'coco' is supported in build_dataloader currently. "
                f"YOLO loader is planned (see BLUEPRINT.md M0)."
            )

        print(f"Dataset: {spec.name} ({spec.format}) — {spec.description}")

        self.train_loader = build_dataloader(
            image_dir=spec.train.images,
            annotation_file=spec.train.annotations,
            input_size=self.model_config.input_size,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            is_training=True,
            mosaic=self.train_config.mosaic,
        )

        self.val_loader = build_dataloader(
            image_dir=spec.val.images,
            annotation_file=spec.val.annotations,
            input_size=self.model_config.input_size,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            is_training=False,
        )

        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

    def train_one_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        self.scheduler.step(epoch)

        total_loss = 0.0
        total_obj_loss = 0.0
        total_bbox_loss = 0.0
        total_positives = 0
        num_batches = 0

        current_lr = self.optimizer.param_groups[0]["lr"]
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.train_config.epochs}")

        for batch in pbar:
            images = batch["images"].to(self.device)
            gt_boxes = batch["boxes"]

            self.optimizer.zero_grad()

            use_amp = self.train_config.amp and self.device.type == "cuda"
            with autocast(enabled=use_amp):
                objectness, bbox_reg, centerness = self.model(images)
                losses = self.criterion(objectness, bbox_reg, centerness, gt_boxes)

            self.scaler.scale(losses["total_loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update EMA after each step
            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += losses["total_loss"].item()
            total_obj_loss += losses["objectness_loss"].item()
            total_bbox_loss += losses["bbox_loss"].item()
            total_positives += losses["num_positives"]
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{losses['total_loss'].item():.4f}",
                "obj": f"{losses['objectness_loss'].item():.4f}",
                "bbox": f"{losses['bbox_loss'].item():.4f}",
                "pos": losses["num_positives"],
                "lr": f"{current_lr:.6f}",
            })

        return {
            "loss": total_loss / max(num_batches, 1),
            "obj_loss": total_obj_loss / max(num_batches, 1),
            "bbox_loss": total_bbox_loss / max(num_batches, 1),
            "positives": total_positives,
            "lr": current_lr,
        }

    def validate(self) -> dict:
        """Validate both raw and (if enabled) EMA models.

        For short training runs on small datasets, EMA often lags badly because
        the running average is still mostly initialization weights. We log both
        and use the raw model's mAP for "best" selection — EMA can only be
        trusted once `ema.updates` is well into the hundreds.
        """
        raw_metrics = evaluate_model(
            self.model, self.val_loader, device=self.device, iou_threshold=0.5,
        )
        out = {f"raw_{k}": v for k, v in raw_metrics.items()}

        if self.ema is not None:
            ema_metrics = evaluate_model(
                self.ema.ema_model, self.val_loader, device=self.device, iou_threshold=0.5,
            )
            out.update({f"ema_{k}": v for k, v in ema_metrics.items()})

        # Primary "mAP50" used by best-checkpoint selection: max(raw, EMA).
        # During warmup EMA is at 0; later it usually beats raw.
        primary_mAP = raw_metrics.get("mAP50", 0.0)
        if self.ema is not None:
            primary_mAP = max(primary_mAP, out.get("ema_mAP50", 0.0))
        out["mAP50"] = primary_mAP
        return out

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        save_dir = self.train_config.save_dir
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "train_config": self.train_config,
            "metrics": metrics,
        }

        # Save EMA weights separately (this is what you deploy)
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.ema_model.state_dict()

        torch.save(checkpoint, os.path.join(save_dir, "latest.pt"))

        if (epoch + 1) % self.train_config.save_interval == 0:
            torch.save(checkpoint, os.path.join(save_dir, f"epoch_{epoch+1}.pt"))

        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, "best.pt"))
            print(f"  ★ New best model (mAP@0.5: {metrics.get('mAP50', 0):.4f})")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint.get("metrics", {}).get("mAP50", 0.0)

        if self.ema and "ema_state_dict" in checkpoint:
            self.ema.ema_model.load_state_dict(checkpoint["ema_state_dict"])

        print(f"Resumed from epoch {self.start_epoch}, best mAP@0.5: {self.best_metric:.4f}")

    def train(self, resume_from: Optional[str] = None):
        """Full training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)

        self.build_dataloaders()

        params = self.model.count_parameters()
        print(f"\n{'='*60}")
        print(f"BoxVision v2 Training")
        print(f"{'='*60}")
        print(f"Backbone:     {self.model_config.backbone}")
        print(f"FPN channels: {self.model_config.fpn_out_channels}")
        print(f"Input size:   {self.model_config.input_size}")
        print(f"Parameters:   {params['total']:,} ({params['total_mb']:.2f} MB)")
        print(f"Device:       {self.device}")
        print(f"Epochs:       {self.train_config.epochs}")
        print(f"Batch size:   {self.train_config.batch_size}")
        print(f"EMA:          {'ON' if self.ema else 'OFF'}")
        print(f"Mosaic:       {'ON' if self.train_config.mosaic else 'OFF'}")
        print(f"{'='*60}\n")

        # Cap the mosaic-off window so short training runs still get augmentation.
        # If mosaic_off_epochs >= total epochs, mosaic would never be on.
        effective_off = min(self.train_config.mosaic_off_epochs, self.train_config.epochs // 2)

        for epoch in range(self.start_epoch, self.train_config.epochs):
            # Mosaic scheduling: disable for last N epochs
            if self.train_config.mosaic:
                remaining = self.train_config.epochs - epoch
                mosaic_on = remaining > effective_off
                self.train_loader.dataset.set_mosaic(mosaic_on)
                if not mosaic_on and remaining == effective_off:
                    print(f"  Mosaic OFF for final {effective_off} epochs")

            train_metrics = self.train_one_epoch(epoch)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Loss: {train_metrics['loss']:.4f} "
                  f"(obj: {train_metrics['obj_loss']:.4f}, "
                  f"bbox: {train_metrics['bbox_loss']:.4f})")
            print(f"  Positives: {train_metrics['positives']}, LR: {train_metrics['lr']:.6f}")

            # Validate periodically
            val_metrics = {}
            if (epoch + 1) % self.train_config.eval_interval == 0:
                print(f"\n  Validation:")
                val_metrics = self.validate()
                print(f"  raw mAP@0.5: {val_metrics.get('raw_mAP50', 0):.4f}  "
                      f"(P={val_metrics.get('raw_precision', 0):.4f} "
                      f"R={val_metrics.get('raw_recall', 0):.4f})")
                if "ema_mAP50" in val_metrics:
                    ema_updates = self.ema.updates if self.ema else 0
                    print(f"  ema mAP@0.5: {val_metrics.get('ema_mAP50', 0):.4f}  "
                          f"(P={val_metrics.get('ema_precision', 0):.4f} "
                          f"R={val_metrics.get('ema_recall', 0):.4f}, "
                          f"updates={ema_updates})")

            is_best = val_metrics.get("mAP50", 0) > self.best_metric
            if is_best:
                self.best_metric = val_metrics["mAP50"]

            all_metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, all_metrics, is_best=is_best)

        print(f"\nTraining complete! Best mAP@0.5: {self.best_metric:.4f}")
        print(f"Best model: {os.path.join(self.train_config.save_dir, 'best.pt')}")
