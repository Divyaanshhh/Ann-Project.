"""Evaluation engine for assessing trained wildfire forecasting models across multiple temporal horizons."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG
from utils.logger import get_logger
from utils.device import get_device
from utils.checkpoint import load_checkpoint
from datasets.data_loader import create_dataloaders
from models.model_factory import build_model
from training.metrics import compute_metrics

logger = get_logger("evaluator")


class FirecastEvaluator:
    """Evaluates trained wildfire progression models per horizon step (12h, 24h, 48h)."""

    def __init__(
        self,
        checkpoint_path: Path | str = CONFIG.checkpoint_dir / "best_model.pt",
        device: torch.device | None = None
    ):
        self.device = device or get_device()
        self.checkpoint_path = Path(checkpoint_path)

        # Build model and load weights
        self.model = build_model().to(self.device)
        self.epoch, self.train_metrics, self.model_config = load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model=self.model,
            device=self.device
        )
        self.model.eval()

    def evaluate(self, test_loader) -> Dict[str, Any]:
        """Runs inference over test dataset and computes overall and per-horizon metrics."""
        logger.info(f"Evaluating model on test dataset ({len(test_loader.dataset)} samples)...")
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                pred = self.model(x)
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        # Shape: (N, 3, 1, 128, 128)
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        # 1. Overall aggregate metrics across all horizons
        overall_metrics = compute_metrics(cat_preds, cat_targets, threshold=CONFIG.training.fire_threshold)

        # 2. Per-horizon metrics
        horizon_labels = [f"+{h}h" for h in CONFIG.data.horizons_hours]
        per_horizon = {}

        for step_idx, label in enumerate(horizon_labels):
            p_step = cat_preds[:, step_idx : step_idx + 1]
            t_step = cat_targets[:, step_idx : step_idx + 1]
            per_horizon[label] = compute_metrics(p_step, t_step, threshold=CONFIG.training.fire_threshold)

        results = {
            "checkpoint": str(self.checkpoint_path.name),
            "checkpoint_epoch": self.epoch,
            "overall_metrics": overall_metrics,
            "per_horizon_metrics": per_horizon
        }

        logger.info("=== Overall Test Metrics ===")
        for k, v in overall_metrics.items():
            logger.info(f"  {k.upper()}: {v}")

        logger.info("=== Per-Horizon Forecast IoU & Dice ===")
        for label, m in per_horizon.items():
            logger.info(f"  {label} -> IoU: {m['iou']:.4f} | Dice: {m['dice']:.4f} | F1: {m['f1']:.4f}")

        # Save to disk
        out_path = CONFIG.checkpoint_dir / "evaluation_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {out_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate FIRECAST AI model on test split.")
    parser.add_argument("--checkpoint", type=str, default=str(CONFIG.checkpoint_dir / "best_model.pt"))
    args = parser.parse_args()

    dataset_file = CONFIG.dataset_path
    _, _, test_loader, _ = create_dataloaders(dataset_path=dataset_file)

    evaluator = FirecastEvaluator(checkpoint_path=args.checkpoint)
    evaluator.evaluate(test_loader)


if __name__ == "__main__":
    main()

