"""Custom loss functions for wildfire progression and intensity forecasting.

Combines:
1. Binary Focal Loss for sparse fire-perimeter segmentation.
2. Masked / Weighted Mean Squared Error (MSE) for continuous fire intensity estimation.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG


class BinaryFocalLoss(nn.Module):
    """Focal Loss for addressing extreme foreground/background class imbalance in wildfire masks.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = CONFIG.training.focal_alpha, gamma: float = CONFIG.training.focal_gamma, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_probs: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_probs: Sigmoid-activated probabilities [0, 1] of shape (B, T, C, H, W).
            target_masks: Binary ground truth {0, 1} of shape (B, T, C, H, W).
        """
        pred_probs = torch.clamp(pred_probs, self.eps, 1.0 - self.eps)

        pt = torch.where(target_masks == 1.0, pred_probs, 1.0 - pred_probs)
        alpha_t = torch.where(target_masks == 1.0, self.alpha, 1.0 - self.alpha)

        focal_weight = alpha_t * torch.pow((1.0 - pt), self.gamma)
        loss = -focal_weight * torch.log(pt)

        return loss.mean()


class IntensityMSELoss(nn.Module):
    """Measures intensity prediction error on continuous fire intensity ground truth.
    
    Focuses supervision on active fire zones to prevent zero-intensity background from dominating.
    """

    def __init__(self, active_weight: float = 2.0):
        super().__init__()
        self.active_weight = active_weight

    def forward(self, pred_intensity: torch.Tensor, target_intensity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_intensity: Predicted intensity tensor (B, T, C, H, W).
            target_intensity: Ground truth intensity tensor (B, T, C, H, W).
        """
        diff_sq = (pred_intensity - target_intensity) ** 2
        # Apply higher weight where fire intensity is present
        mask = (target_intensity > 0.05).float()
        weights = 1.0 + (self.active_weight - 1.0) * mask
        weighted_loss = diff_sq * weights
        return weighted_loss.mean()


class FirecastCompoundLoss(nn.Module):
    """Total Compound Loss = focal_weight * Focal_Loss + intensity_weight * Intensity_MSE."""

    def __init__(
        self,
        focal_weight: float = CONFIG.training.focal_weight,
        intensity_weight: float = CONFIG.training.intensity_weight,
        alpha: float = CONFIG.training.focal_alpha,
        gamma: float = CONFIG.training.focal_gamma
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.intensity_weight = intensity_weight
        self.focal_loss = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.intensity_loss = IntensityMSELoss()

    def forward(
        self,
        pred_presence: torch.Tensor,
        target_presence: torch.Tensor,
        pred_intensity: torch.Tensor | None = None,
        target_intensity: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes total loss and itemized dictionary."""
        # Binarize presence target for focal loss
        binary_target = (target_presence > 0.2).float()
        l_focal = self.focal_loss(pred_presence, binary_target)

        loss_dict = {"focal_loss": float(l_focal.item())}
        total = self.focal_weight * l_focal

        if pred_intensity is not None and target_intensity is not None:
            l_mse = self.intensity_loss(pred_intensity, target_intensity)
            total = total + self.intensity_weight * l_mse
            loss_dict["intensity_mse"] = float(l_mse.item())

        loss_dict["total_loss"] = float(total.item())
        return total, loss_dict

