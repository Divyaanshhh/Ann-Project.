"""Tensor dimensionality and numerical integrity validation for FIRECAST AI."""

from typing import Tuple
import numpy as np
import torch


def validate_input_tensor(
    tensor: torch.Tensor | np.ndarray,
    expected_channels: int = 6,
    expected_seq_len: int = 5,
    spatial_size: Tuple[int, int] = (128, 128)
) -> None:
    """Validates that spatiotemporal input tensor strictly follows (B, T, C, H, W).
    
    Args:
        tensor: Input PyTorch Tensor or NumPy array.
        expected_channels: Expected channel dimension (default 6).
        expected_seq_len: Expected history observation sequence length (default 5).
        spatial_size: Expected (H, W) tuple (default (128, 128)).
        
    Raises:
        ValueError: If shape, types, or numerical validity checks fail.
    """
    if not isinstance(tensor, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Input must be a torch.Tensor or np.ndarray, got {type(tensor)}")

    shape = tensor.shape
    if len(shape) != 5:
        raise ValueError(
            f"Expected 5D tensor (Batch, Time, Channels, Height, Width), but got shape {shape}"
        )

    b, t, c, h, w = shape
    if t != expected_seq_len:
        raise ValueError(f"Expected time dimension {expected_seq_len}, got {t} in shape {shape}")
    if c != expected_channels:
        raise ValueError(f"Expected channel dimension {expected_channels}, got {c} in shape {shape}")
    if (h, w) != spatial_size:
        raise ValueError(f"Expected spatial resolution {spatial_size}, got ({h}, {w}) in shape {shape}")

    # Numerical validity
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any():
            raise ValueError("Input tensor contains NaN values.")
        if torch.isinf(tensor).any():
            raise ValueError("Input tensor contains infinite values.")
    else:
        if np.isnan(tensor).any():
            raise ValueError("Input array contains NaN values.")
        if np.isinf(tensor).any():
            raise ValueError("Input array contains infinite values.")


def validate_target_tensor(
    tensor: torch.Tensor | np.ndarray,
    expected_channels: int = 1,
    expected_pred_len: int = 3,
    spatial_size: Tuple[int, int] = (128, 128)
) -> None:
    """Validates that spatiotemporal target tensor strictly follows (B, T_pred, C_out, H, W).
    
    Args:
        tensor: Target PyTorch Tensor or NumPy array.
        expected_channels: Expected target channels (default 1 for fire presence).
        expected_pred_len: Expected future steps (default 3 for 12h, 24h, 48h).
        spatial_size: Expected (H, W) tuple (default (128, 128)).
    """
    if not isinstance(tensor, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Target must be a torch.Tensor or np.ndarray, got {type(tensor)}")

    shape = tensor.shape
    if len(shape) != 5:
        raise ValueError(
            f"Expected 5D target tensor (Batch, Pred_Time, Channels, Height, Width), got {shape}"
        )

    b, t, c, h, w = shape
    if t != expected_pred_len:
        raise ValueError(f"Expected prediction time dimension {expected_pred_len}, got {t}")
    if c != expected_channels:
        raise ValueError(f"Expected target channel {expected_channels}, got {c}")
    if (h, w) != spatial_size:
        raise ValueError(f"Expected spatial resolution {spatial_size}, got ({h}, {w})")


def sanitize_nans(tensor: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Safely replaces NaNs and Infs with a fallback value."""
    tensor = torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return tensor

