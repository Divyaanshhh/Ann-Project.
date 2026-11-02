"""Synthetic Spatio-Temporal Wildfire Dataset Generator.

Simulates physically inspired wildfire progression across realistic geospatial layers:
- Fuel density (NDVI)
- Topography (DEM slope and aspect)
- Dynamic meteorological forcing (wind speed, wind direction, atmospheric dryness)
- Spatially correlated fire front advection and intensity dispersion.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import scipy.ndimage as ndimage
import torch

# Ensure local imports work when executed via `python -m datasets.synthetic_generator`
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG
from utils.logger import get_logger
from utils.validation import validate_input_tensor, validate_target_tensor

logger = get_logger("synthetic_generator")


class SyntheticWildfireGenerator:
    """Generates synthetic multi-modal wildfire datasets with physically motivated spatiotemporal continuity."""

    def __init__(
        self,
        num_samples: int = 100,
        seq_len: int = 5,
        pred_len: int = 3,
        height: int = 128,
        width: int = 128,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_steps = seq_len + pred_len  # 8 total consecutive timesteps
        self.height = height
        self.width = width
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _generate_spatial_field(self, scale: float = 12.0, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        """Generates a smooth, spatially correlated random field using Gaussian filtering."""
        noise = self.rng.standard_normal((self.height, self.width))
        smoothed = ndimage.gaussian_filter(noise, sigma=scale)
        # Min-max scale to [low, high]
        s_min, s_max = smoothed.min(), smoothed.max()
        if s_max - s_min > 1e-6:
            normalized = (smoothed - s_min) / (s_max - s_min)
        else:
            normalized = np.full_like(smoothed, 0.5)
        return low + normalized * (high - low)

    def generate_single_sample(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates one multi-temporal sample.
        
        Returns:
            input_sequence: (seq_len=5, 6, 128, 128)
            target_sequence: (pred_len=3, 1, 128, 128)
        """
        # 1. Base Geospatial Static Layers
        # NDVI fuel map [0.1, 0.9]
        ndvi_base = self._generate_spatial_field(scale=16.0, low=0.15, high=0.85)
        # Terrain DEM risk (slope/aspect combination) [0.1, 0.95]
        terrain_risk = self._generate_spatial_field(scale=20.0, low=0.1, high=0.9)

        # 2. Base Weather dynamics
        # Wind velocity (m/s normalized)
        base_wind_speed = float(self.rng.uniform(0.2, 0.85))
        # Wind direction in degrees (0 - 360)
        base_wind_dir_deg = float(self.rng.uniform(0, 360))
        wind_dir_rad = np.deg2rad(base_wind_dir_deg)
        # Wind drift vectors (dx, dy)
        wind_dx = np.cos(wind_dir_rad)
        wind_dy = np.sin(wind_dir_rad)

        # Base atmospheric dryness [0.3, 0.95]
        base_dryness = float(self.rng.uniform(0.35, 0.95))

        # 3. Initial Ignition points (1 to 3 clusters)
        num_ignitions = self.rng.integers(1, 4)
        fire_state = np.zeros((self.height, self.width), dtype=np.float32)
        for _ in range(num_ignitions):
            cx = self.rng.integers(int(self.width * 0.25), int(self.width * 0.75))
            cy = self.rng.integers(int(self.height * 0.25), int(self.height * 0.75))
            radius = self.rng.integers(3, 7)
            y, x = np.ogrid[:self.height, :self.width]
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            fire_state += np.exp(-dist_sq / (2.0 * (radius ** 2))).astype(np.float32)

        fire_state = np.clip(fire_state, 0.0, 1.0)

        # Simulate all timesteps (seq_len + pred_len)
        full_time_series_inputs = []
        full_time_series_targets = []

        current_fire = fire_state.copy()

        for t in range(self.total_steps):
            # Dynamic weather perturbation over time
            t_wind_speed = np.clip(base_wind_speed + self.rng.normal(0, 0.03), 0.05, 1.0)
            t_wind_dir_deg = (base_wind_dir_deg + self.rng.normal(0, 5.0)) % 360.0
            t_wind_dir_norm = t_wind_dir_deg / 360.0
            t_dryness = np.clip(base_dryness + self.rng.normal(0, 0.02), 0.1, 1.0)

            # Fire Spread Cellular Advection
            # Spread rate depends on: Fuel (NDVI) * Terrain * Wind * Dryness
            spread_kernel = np.array([
                [0.05, 0.10, 0.05],
                [0.10, 0.40, 0.10],
                [0.05, 0.10, 0.05]
            ], dtype=np.float32)

            # Shift kernel center in direction of wind
            shift_x = int(np.round(wind_dx * t_wind_speed * 1.8))
            shift_y = int(np.round(wind_dy * t_wind_speed * 1.8))

            diffused_fire = ndimage.convolve(current_fire, spread_kernel, mode='constant', cval=0.0)
            shifted_fire = ndimage.shift(diffused_fire, shift=(shift_y, shift_x), mode='nearest')

            # Environmental acceleration
            environment_factor = (
                0.35 * ndvi_base +
                0.25 * terrain_risk +
                0.25 * t_dryness +
                0.15 * t_wind_speed
            )

            # Growth & thresholding
            next_fire = current_fire * 0.85 + shifted_fire * (0.35 + 0.5 * environment_factor)
            # Add small random spot embers near front
            active_mask = next_fire > 0.25
            if active_mask.any() and self.rng.random() < 0.3:
                noise_spots = (self.rng.random((self.height, self.width)) > 0.995).astype(np.float32)
                next_fire += noise_spots * active_mask * 0.4

            next_fire = np.clip(next_fire, 0.0, 1.0)
            current_fire = next_fire

            # Input channels for this time step (6 channels)
            # Channel 0: Fire perimeter and intensity
            # Channel 1: NDVI (slightly reduced where burned)
            c1_ndvi = np.clip(ndvi_base - 0.4 * current_fire, 0.0, 1.0)
            # Channel 2: Terrain DEM risk
            c2_dem = terrain_risk
            # Channel 3: Wind speed field
            c3_ws = np.full((self.height, self.width), t_wind_speed, dtype=np.float32)
            # Channel 4: Wind direction field (normalized circular [0, 1])
            c4_wd = np.full((self.height, self.width), t_wind_dir_norm, dtype=np.float32)
            # Channel 5: Atmospheric dryness field
            c5_dry = np.full((self.height, self.width), t_dryness, dtype=np.float32)

            step_6ch = np.stack([current_fire, c1_ndvi, c2_dem, c3_ws, c4_wd, c5_dry], axis=0) # (6, H, W)

            if t < self.seq_len:
                full_time_series_inputs.append(step_6ch)
            else:
                # Target: binary/probabilistic fire mask (1, H, W)
                target_frame = current_fire[np.newaxis, :, :].copy()
                full_time_series_targets.append(target_frame)

        # Stack into (T_in=5, 6, 128, 128) and (T_out=3, 1, 128, 128)
        input_array = np.stack(full_time_series_inputs, axis=0).astype(np.float32)
        target_array = np.stack(full_time_series_targets, axis=0).astype(np.float32)

        return input_array, target_array

    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates all synthetic samples.
        
        Returns:
            X: (N, 5, 6, 128, 128)
            Y: (N, 3, 1, 128, 128)
        """
        logger.info(f"Generating {self.num_samples} synthetic wildfire spatiotemporal samples...")
        inputs_list = []
        targets_list = []

        for i in range(self.num_samples):
            x_i, y_i = self.generate_single_sample(i)
            inputs_list.append(x_i)
            targets_list.append(y_i)

        X = np.stack(inputs_list, axis=0)
        Y = np.stack(targets_list, axis=0)

        logger.info(f"Generated synthetic inputs shape: {X.shape}, dtype: {X.dtype}")
        logger.info(f"Generated synthetic targets shape: {Y.shape}, dtype: {Y.dtype}")

        # Validate dimensional constraints strictly
        validate_input_tensor(X, expected_channels=6, expected_seq_len=self.seq_len)
        validate_target_tensor(Y, expected_channels=1, expected_pred_len=self.pred_len)

        return X, Y

    def save_dataset(self, output_path: Path | str) -> Path:
        """Generates and saves the synthetic dataset to a compressed .npz archive."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        X, Y = self.generate_dataset()
        np.savez_compressed(
            output_path,
            inputs=X,
            targets=Y,
            seed=self.seed,
            num_samples=self.num_samples,
            spatial_resolution="128x128",
            channels=CONFIG.channels.names
        )
        logger.info(f"Successfully saved synthetic wildfire dataset to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic wildfire dataset for FIRECAST AI.")
    parser.add_argument("--samples", type=int, default=CONFIG.data.num_synthetic_samples, help="Number of samples")
    parser.add_argument("--output", type=str, default=str(CONFIG.synthetic_dir / "synthetic_wildfire_dataset.npz"))
    parser.add_argument("--seed", type=int, default=CONFIG.data.seed)
    args = parser.parse_args()

    generator = SyntheticWildfireGenerator(
        num_samples=args.samples,
        seq_len=CONFIG.data.history_steps,
        pred_len=CONFIG.data.forecast_steps,
        height=CONFIG.data.spatial_height,
        width=CONFIG.data.spatial_width,
        seed=args.seed
    )
    generator.save_dataset(args.output)


if __name__ == "__main__":
    main()
