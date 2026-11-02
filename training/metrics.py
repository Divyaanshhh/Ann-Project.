"""Evaluation metrics for wildfire progression and intensity forecasting."""

from typing import Dict
import torch
import torch.nn.functional as F

from config import CONFIG


def compute_metrics(
    pred_probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = CONFIG.training.fire_threshold,
    eps: float = 1e-7
) -> Dict[str, float]:
    """Calculates comprehensive spatial-temporal classification & segmentation metrics.
    
    Args:
        pred_probs: Model predicted probabilities in [0, 1] of shape (B, T, C, H, W).
        targets: Target masks/values in [0, 1] of shape (B, T, C, H, W).
        threshold: Fire classification decision threshold (default 0.5).
        eps: Small epsilon to avoid division by zero.
        
    Returns:
        Dictionary containing IoU, Dice, Precision, Recall, F1, BCE, and MAE.
    """
    with torch.no_grad():
        # Ensure tensors are float and detached
        pred = pred_probs.detach()
        tgt = targets.detach()

        # Binary ground truth & binary prediction
        tgt_bin = (tgt > 0.2).float()
        pred_bin = (pred >= threshold).float()

        # Flatten spatial & temporal dimensions for global metrics
        p_flat = pred_bin.view(-1)
        t_flat = tgt_bin.view(-1)

        # True Positives, False Positives, False Negatives
        tp = (p_flat * t_flat).sum().item()
        fp = (p_flat * (1.0 - t_flat)).sum().item()
        fn = ((1.0 - p_flat) * t_flat).sum().item()

        # Precision, Recall, F1
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)

        # Intersection over Union (Jaccard Index)
        intersection = tp
        union = tp + fp + fn
        iou = (intersection + eps) / (union + eps)

        # Dice Coefficient (Sørensen–Dice)
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

        # Binary Cross Entropy
        bce = F.binary_cross_entropy(torch.clamp(pred, 1e-7, 1.0 - 1e-7), tgt_bin).item()

        # Intensity Mean Absolute Error (MAE)
        mae = torch.abs(pred - tgt).mean().item()

        return {
            "iou": round(float(iou), 4),
            "dice": round(float(dice), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "bce": round(float(bce), 4),
            "mae": round(float(mae), 4),
        }

