"""Automated tests for deep learning model architectures, layers, and factories."""

import pytest
import torch

from config import CONFIG
from models.convlstm_cell import ConvLSTMCell
from models.convlstm_model import ConvLSTMForecaster
from models.spatiotemporal_unet import SpatioTemporalUNet
from models.model_factory import build_model


def test_convlstm_cell():
    """Test single ConvLSTM cell state propagation."""
    cell = ConvLSTMCell(in_channels=6, hidden_dim=32, kernel_size=3)
    x = torch.randn(2, 6, 64, 64)
    h0, c0 = cell.init_hidden(batch_size=2, height=64, width=64, device=x.device)

    h1, c1 = cell(x, (h0, c0))
    assert h1.shape == (2, 32, 64, 64)
    assert c1.shape == (2, 32, 64, 64)


def test_convlstm_forecaster_forward():
    """Test full multi-layer ConvLSTM model forward pass and shape preservation."""
    model = ConvLSTMForecaster(
        in_channels=6,
        hidden_dims=[16, 32],
        out_channels=1,
        pred_len=3,
        use_intensity_head=True
    )
    x = torch.randn(2, 5, 6, 128, 128)

    presence, intensity = model(x, return_intensity=True)
    assert presence.shape == (2, 3, 1, 128, 128)
    assert intensity.shape == (2, 3, 1, 128, 128)
    # Sigmoid probability bounded [0, 1]
    assert presence.min() >= 0.0 and presence.max() <= 1.0


def test_spatiotemporal_unet():
    """Test alternative SpatioTemporalUNet forward pass."""
    model = SpatioTemporalUNet(in_channels=6, seq_len=5, pred_len=3, out_channels=1, base_features=16)
    x = torch.randn(2, 5, 6, 128, 128)

    out = model(x)
    assert out.shape == (2, 3, 1, 128, 128)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_model_factory():
    """Test model factory instantiation."""
    m_conv = build_model("convlstm")
    assert isinstance(m_conv, ConvLSTMForecaster)

    m_unet = build_model("spatiotemporal_unet")
    assert isinstance(m_unet, SpatioTemporalUNet)
