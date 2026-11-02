"""Automated tests for autoregressive feedback, uncertainty estimation, and checkpointing."""

import pytest
import numpy as np
import torch

from config import CONFIG
from utils.checkpoint import save_checkpoint, load_checkpoint
from models.convlstm_model import ConvLSTMForecaster
from forecasting.autoregressive import AutoregressiveFeedbackAdapter
from forecasting.uncertainty import MonteCarloUncertaintyEstimator


def test_autoregressive_adapter():
    """Verify single-channel prediction is converted back to valid 6-channel representation."""
    adapter = AutoregressiveFeedbackAdapter()
    pred_fire = torch.rand(2, 1, 128, 128)
    prev_6ch = torch.rand(2, 6, 128, 128)

    next_6ch = adapter.adapt_prediction_to_input_step(pred_fire, prev_6ch)
    assert next_6ch.shape == (2, 6, 128, 128)
    # Check that fire channel matches prediction
    assert torch.allclose(next_6ch[:, 0:1], pred_fire)


def test_monte_carlo_uncertainty():
    """Verify MC dropout generates valid mean, variance, and uncertainty flags."""
    model = ConvLSTMForecaster(in_channels=6, hidden_dims=[16, 32], out_channels=1, pred_len=3, dropout=0.2)
    estimator = MonteCarloUncertaintyEstimator(num_samples=3)

    x = torch.randn(1, 5, 6, 64, 64)
    unc = estimator.estimate_uncertainty(model, x)

    assert "mean_probability" in unc
    assert "variance" in unc
    assert "std_dev" in unc
    assert unc["std_dev"].shape == (1, 3, 1, 64, 64)


def test_checkpoint_saving_and_restoration(tmp_path):
    """Verify checkpoint saving and exact parameter restoration."""
    model = ConvLSTMForecaster(in_channels=6, hidden_dims=[16, 32], out_channels=1)
    ckpt_file = save_checkpoint(
        model=model,
        epoch=5,
        metrics={"iou": 0.82},
        config_dict={"test": True},
        checkpoint_dir=tmp_path,
        filename="test_ckpt.pt"
    )
    assert ckpt_file.exists()

    new_model = ConvLSTMForecaster(in_channels=6, hidden_dims=[16, 32], out_channels=1)
    epoch, metrics, cfg = load_checkpoint(ckpt_file, new_model)

    assert epoch == 5
    assert metrics["iou"] == 0.82
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)

