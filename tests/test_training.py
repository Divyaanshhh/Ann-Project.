"""Automated tests for loss functions, evaluation metrics, and backward passes."""

import pytest
import torch

from training.losses import BinaryFocalLoss, IntensityMSELoss, FirecastCompoundLoss
from training.metrics import compute_metrics
from models.convlstm_model import ConvLSTMForecaster


def test_binary_focal_loss():
    """Verify focal loss handles class imbalance and outputs positive scalar."""
    focal = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    pred = torch.tensor([[[[[0.1, 0.9], [0.2, 0.8]]]]], dtype=torch.float32)
    tgt = torch.tensor([[[[[0.0, 1.0], [0.0, 1.0]]]]], dtype=torch.float32)

    loss = focal(pred, tgt)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_compound_loss_backward():
    """Verify compound loss gradients propagate through ConvLSTM parameters."""
    model = ConvLSTMForecaster(in_channels=6, hidden_dims=[16, 32], out_channels=1, pred_len=3)
    criterion = FirecastCompoundLoss()

    x = torch.randn(1, 5, 6, 64, 64)
    y = torch.zeros(1, 3, 1, 64, 64)
    y[:, :, :, 20:40, 20:40] = 1.0

    pred_presence, pred_intensity = model(x, return_intensity=True)
    loss, loss_dict = criterion(pred_presence, y, pred_intensity, y)

    loss.backward()

    # Check that model parameters received gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grads, "Model parameters did not receive gradients during backward pass."
    assert "total_loss" in loss_dict
    assert "focal_loss" in loss_dict


def test_metric_computation():
    """Verify calculation of IoU, Dice, Precision, Recall, and F1."""
    pred = torch.zeros(1, 1, 1, 10, 10)
    tgt = torch.zeros(1, 1, 1, 10, 10)

    # 4 active pixels overlap
    pred[0, 0, 0, :4, :4] = 0.8
    tgt[0, 0, 0, :4, :4] = 1.0

    m = compute_metrics(pred, tgt, threshold=0.5)
    assert m["iou"] > 0.9
    assert m["dice"] > 0.9
    assert m["precision"] > 0.9
    assert m["recall"] > 0.9
