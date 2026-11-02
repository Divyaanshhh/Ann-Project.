"""Automated tests for dataset generation, dimensionality, and multi-modal loading."""

import pytest
import numpy as np
import torch

from config import CONFIG
from datasets.synthetic_generator import SyntheticWildfireGenerator
from datasets.data_loader import WildfireDataset, create_dataloaders
from datasets.normalization import SpatioTemporalNormalizer
from datasets.real_wildfire_adapter import RealWildfireAdapter
from utils.validation import validate_input_tensor, validate_target_tensor


def test_synthetic_generator_dimensions():
    """Verify generator produces exact (N, 5, 6, 128, 128) inputs and (N, 3, 1, 128, 128) targets."""
    gen = SyntheticWildfireGenerator(num_samples=4, height=128, width=128, seed=99)
    x, y = gen.generate_dataset()

    assert x.shape == (4, 5, 6, 128, 128), f"Unexpected input shape: {x.shape}"
    assert y.shape == (4, 3, 1, 128, 128), f"Unexpected target shape: {y.shape}"
    assert x.dtype == np.float32
    assert y.dtype == np.float32
    validate_input_tensor(x)
    validate_target_tensor(y)


def test_normalizer_bounds():
    """Test channel-wise normalization keeps values bounded in [0, 1]."""
    normalizer = SpatioTemporalNormalizer()
    dummy_tensor = torch.randn(2, 5, 6, 128, 128)
    normed = normalizer.normalize_tensor(dummy_tensor)

    assert normed.min() >= 0.0
    assert normed.max() <= 1.0


def test_wildfire_dataset_and_augmentation():
    """Test PyTorch WildfireDataset indexing and tensor shapes."""
    inputs = np.random.uniform(0.0, 1.0, (6, 5, 6, 128, 128)).astype(np.float32)
    targets = np.random.uniform(0.0, 1.0, (6, 3, 1, 128, 128)).astype(np.float32)

    ds = WildfireDataset(inputs, targets, augment=True)
    assert len(ds) == 6

    x_item, y_item = ds[0]
    assert x_item.shape == (5, 6, 128, 128)
    assert y_item.shape == (3, 1, 128, 128)
    assert isinstance(x_item, torch.Tensor)
    assert isinstance(y_item, torch.Tensor)


def test_real_wildfire_adapter():
    """Test real NASA FIRMS data adapter rasterization."""
    adapter = RealWildfireAdapter()
    df = adapter.fetch_or_load_real_firms_data()
    assert len(df) > 0
    assert "latitude" in df.columns
    assert "longitude" in df.columns

    x, y = adapter.rasterize_firms_cluster(df)
    assert x.shape == (5, 6, 128, 128)
    assert y.shape == (3, 1, 128, 128)

