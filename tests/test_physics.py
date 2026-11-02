"""Automated tests for Physics-Informed Neural Network (PINN) wildfire spread loss."""

import pytest
import torch
from training.physics_loss import PhysicsInformedSpreadLoss
from training.losses import FirecastCompoundLoss


def test_pinn_loss_gradients():
    """Verify PINN loss produces positive scalar and valid gradients."""
    pinn = PhysicsInformedSpreadLoss()

    # Pred: (B=2, T=3, C=1, H=64, W=64)
    pred = torch.rand(2, 3, 1, 64, 64, requires_grad=True)
    # Input 6ch: (B=2, T=5, C=6, H=64, W=64)
    input_6ch = torch.rand(2, 5, 6, 64, 64)

    loss, meta = pinn(pred, input_6ch)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert "pinn_loss" in meta
    assert "mean_advection" in meta

    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_compound_loss_with_physics():
    """Verify compound loss incorporates physics loss when input_6ch is provided."""
    criterion = FirecastCompoundLoss(physics_weight=0.3)

    pred = torch.rand(1, 3, 1, 32, 32)
    target = torch.zeros(1, 3, 1, 32, 32)
    input_6ch = torch.rand(1, 5, 6, 32, 32)

    total_loss, loss_dict = criterion(pred, target, input_6ch=input_6ch)
    assert "pinn_loss" in loss_dict
    assert total_loss.item() > 0.0
