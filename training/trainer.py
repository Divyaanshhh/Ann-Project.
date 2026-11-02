"""Model training engine with GPU acceleration, compound loss, and early stopping."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Enable local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG
from utils.logger import get_logger
from utils.device import get_device, clear_gpu_cache
from utils.checkpoint import save_checkpoint
from datasets.data_loader import create_dataloaders
from datasets.normalization import SpatioTemporalNormalizer
from models.model_factory import build_model
from training.losses import FirecastCompoundLoss
from training.metrics import compute_metrics

logger = get_logger("trainer")


class FirecastTrainer:
    """Orchestrates end-to-end model training, validation, checkpointing, and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config=CONFIG,
        device: torch.device | None = None
    ):
        self.config = config
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer & Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.lr_min
        )

        # Loss function
        self.criterion = FirecastCompoundLoss(
            focal_weight=config.training.focal_weight,
            intensity_weight=config.training.intensity_weight,
            alpha=config.training.focal_alpha,
            gamma=config.training.focal_gamma
        )

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "lr": []
        }

    def train_epoch(self, epoch: int) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.config.model.use_intensity_head:
                pred_presence, pred_intensity = self.model(x, return_intensity=True)
                loss, _ = self.criterion(
                    pred_presence=pred_presence,
                    target_presence=y,
                    pred_intensity=pred_intensity,
                    target_intensity=y
                )
            else:
                pred_presence = self.model(x)
                loss, _ = self.criterion(pred_presence=pred_presence, target_presence=y)

            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.gradient_clip_val
            )

            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Runs a validation epoch and calculates multi-metric evaluation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if self.config.model.use_intensity_head:
                    pred_presence, pred_intensity = self.model(x, return_intensity=True)
                    loss, _ = self.criterion(
                        pred_presence=pred_presence,
                        target_presence=y,
                        pred_intensity=pred_intensity,
                        target_intensity=y
                    )
                else:
                    pred_presence = self.model(x)
                    loss, _ = self.criterion(pred_presence=pred_presence, target_presence=y)

                total_loss += loss.item()
                all_preds.append(pred_presence.cpu())
                all_targets.append(y.cpu())

        avg_loss = total_loss / max(num_batches, 1)

        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(cat_preds, cat_targets, threshold=self.config.training.fire_threshold)

        return avg_loss, metrics

    def train(self, num_epochs: int | None = None) -> Dict[str, Any]:
        """Executes full training loop with early stopping and checkpointing."""
        epochs = num_epochs or self.config.training.num_epochs
        logger.info(f"Starting training for {epochs} epochs on device: {self.device}...")

        best_val_iou = -1.0
        patience_counter = 0
        best_epoch = 0

        # Save normalization parameters
        normalizer = SpatioTemporalNormalizer()
        norm_path = self.config.checkpoint_dir / "normalization_params.json"
        normalizer.save(norm_path)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate_epoch()
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_loss, 5))
            self.history["val_loss"].append(round(val_loss, 5))
            self.history["val_iou"].append(metrics["iou"])
            self.history["val_dice"].append(metrics["dice"])
            self.history["val_precision"].append(metrics["precision"])
            self.history["val_recall"].append(metrics["recall"])
            self.history["val_f1"].append(metrics["f1"])
            self.history["lr"].append(current_lr)

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val IoU: {metrics['iou']:.4f} | "
                f"Dice: {metrics['dice']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

            # Checkpoint best model based on validation IoU
            is_best = metrics["iou"] > best_val_iou
            if is_best:
                best_val_iou = metrics["iou"]
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    model=self.model,
                    epoch=epoch,
                    metrics=metrics,
                    config_dict={"model_type": self.config.model.model_type, "hidden_dims": self.config.model.hidden_dims},
                    checkpoint_dir=self.config.checkpoint_dir,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    is_best=True
                )
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (no IoU improvement for {patience_counter} epochs).")
                break

        # Save final model checkpoint
        save_checkpoint(
            model=self.model,
            epoch=epoch,
            metrics=metrics,
            config_dict={"model_type": self.config.model.model_type, "hidden_dims": self.config.model.hidden_dims},
            checkpoint_dir=self.config.checkpoint_dir,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            is_best=False,
            filename="final_model.pt"
        )

        # Save training history JSON
        history_path = self.config.checkpoint_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        clear_gpu_cache()
        logger.info(f"Training completed. Best validation IoU: {best_val_iou:.4f} at epoch {best_epoch}.")
        return self.history


def main():
    parser = argparse.ArgumentParser(description="Train FIRECAST AI Wildfire Progression Model.")
    parser.add_argument("--epochs", type=int, default=CONFIG.training.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=CONFIG.data.batch_size, help="Batch size")
    parser.add_argument("--model-type", type=str, default=CONFIG.model.model_type, help="Model type: convlstm or spatiotemporal_unet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()

    # Reproducibility seed
    torch.manual_seed(CONFIG.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.data.seed)

    # Ensure synthetic dataset is generated
    dataset_file = CONFIG.synthetic_dir / "synthetic_wildfire_dataset.npz"
    if not dataset_file.exists():
        from datasets.synthetic_generator import SyntheticWildfireGenerator
        gen = SyntheticWildfireGenerator()
        gen.save_dataset(dataset_file)

    train_loader, val_loader, _, _ = create_dataloaders(
        dataset_path=dataset_file,
        batch_size=args.batch_size
    )

    device = torch.device(args.device)
    model = build_model(model_type=args.model_type)

    trainer = FirecastTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
