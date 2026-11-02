"""Research-oriented Monte Carlo Dropout uncertainty estimation engine."""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np


class MonteCarloUncertaintyEstimator:
    """Estimates epistemic forecasting uncertainty through Monte Carlo Dropout sampling."""

    def __init__(self, num_samples: int = 10, uncertainty_threshold: float = 0.15):
        self.num_samples = num_samples
        self.uncertainty_threshold = uncertainty_threshold

    def enable_dropout_only(self, model: nn.Module) -> None:
        """Enables dropout layers while keeping Batch/GroupNorm in eval mode."""
        model.eval()
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def estimate_uncertainty(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor | Any]:
        """Runs M stochastic forward passes and computes mean, variance, and uncertainty flags.
        
        Args:
            model: Trained model with dropout.
            x: Input tensor (B, 5, 6, 128, 128).
            
        Returns:
            Dictionary containing:
                mean_probability: (B, 3, 1, 128, 128)
                variance: (B, 3, 1, 128, 128)
                std_dev: (B, 3, 1, 128, 128)
                high_uncertainty_mask: Boolean tensor of high variance areas
                samples: List of all M stochastic outputs
        """
        self.enable_dropout_only(model)

        mc_predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                mc_predictions.append(pred)

        # Shape: (M, B, 3, 1, 128, 128)
        stacked = torch.stack(mc_predictions, dim=0)

        mean_prob = torch.mean(stacked, dim=0)
        variance = torch.var(stacked, dim=0)
        std_dev = torch.sqrt(variance + 1e-7)

        # Identify pixels where variance exceeds threshold
        high_uncertainty_mask = (std_dev > self.uncertainty_threshold)

        return {
            "mean_probability": mean_prob,
            "variance": variance,
            "std_dev": std_dev,
            "high_uncertainty_mask": high_uncertainty_mask,
            "num_mc_samples": self.num_samples,
            "disclaimer": (
                "Academic Research Output: Model confidence and uncertainty are experimental estimates "
                "derived via Monte Carlo dropout and do NOT represent authoritative emergency guarantees."
            )
        }

