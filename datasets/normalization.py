"""Normalization and feature scaling engine for 6-channel spatiotemporal tensors."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch


class SpatioTemporalNormalizer:
    """Handles channel-wise normalization, circular angle transforms, and scaling.
    
    Channels:
        0: Fire perimeter and normalized intensity [0, 1]
        1: NDVI vegetation index (originally [-1, 1], mapped to [0, 1])
        2: Terrain risk (DEM slope + aspect composite, [0, 1])
        3: Wind speed (0 to 100 km/h, normalized [0, 1])
        4: Wind direction (circular radians / degrees mapped cyclic to [0, 1])
        5: Atmospheric dryness (composite of RH [0-100%] and Temp [-10 to 50°C])
    """

    def __init__(self, stats: Optional[Dict[str, Any]] = None):
        self.stats = stats or {
            "c0_fire": {"min": 0.0, "max": 1.0},
            "c1_ndvi": {"min": -1.0, "max": 1.0},
            "c2_terrain": {"min": 0.0, "max": 1.0},
            "c3_wind_speed": {"min": 0.0, "max": 35.0}, # m/s or scaled
            "c4_wind_dir": {"min": 0.0, "max": 360.0},
            "c5_dryness": {"min": 0.0, "max": 1.0},
        }

    @staticmethod
    def normalize_wind_direction(degrees: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Circularly normalizes wind direction angle (0 to 360 deg) into [0, 1].
        
        Preserves cyclic continuity where 0° == 360°.
        Uses (angle % 360) / 360.0.
        """
        if isinstance(degrees, torch.Tensor):
            return (degrees % 360.0) / 360.0
        return (degrees % 360.0) / 360.0

    @staticmethod
    def normalize_ndvi(ndvi_raw: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Transforms raw NDVI [-1, 1] into [0, 1] range: (ndvi + 1.0) / 2.0."""
        if isinstance(ndvi_raw, torch.Tensor):
            return torch.clamp((ndvi_raw + 1.0) / 2.0, 0.0, 1.0)
        return np.clip((ndvi_raw + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def compute_atmospheric_dryness(
        temperature_c: np.ndarray | torch.Tensor,
        relative_humidity_pct: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Calculates normalized atmospheric dryness index [0, 1] from Temp and RH.
        
        High temperature + low humidity -> High fire risk / dryness (~1.0).
        Low temperature + high humidity -> Low dryness (~0.0).
        """
        # Normalized temperature [-10C to 50C] -> [0, 1]
        t_norm = (temperature_c - (-10.0)) / (50.0 - (-10.0))
        # Inverted humidity [0 to 100%] -> [0, 1] (0% RH is max dryness = 1)
        rh_inv = 1.0 - (relative_humidity_pct / 100.0)

        dryness = 0.5 * t_norm + 0.5 * rh_inv
        if isinstance(dryness, torch.Tensor):
            return torch.clamp(dryness, 0.0, 1.0)
        return np.clip(dryness, 0.0, 1.0)

    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes an entire 6-channel spatiotemporal tensor (B, T, C=6, H, W)."""
        norm_t = tensor.clone()
        # Ensure all channels stay within legitimate physical bound [0, 1]
        return torch.clamp(norm_t, 0.0, 1.0)

    def save(self, filepath: Path | str) -> None:
        """Persists normalization parameters to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    @classmethod
    def load(cls, filepath: Path | str) -> "SpatioTemporalNormalizer":
        """Loads normalization parameters from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls(stats=stats)
