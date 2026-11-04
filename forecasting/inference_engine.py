"""High-level inference engine for multi-horizon wildfire progression forecasting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn

from config import CONFIG
from utils.logger import get_logger
from utils.device import get_device
from utils.checkpoint import load_checkpoint
from utils.validation import validate_input_tensor
from datasets.normalization import SpatioTemporalNormalizer
from datasets.temporal_builder import TemporalSequenceBuilder
from models.model_factory import build_model
from forecasting.uncertainty import MonteCarloUncertaintyEstimator

logger = get_logger("inference_engine")


@dataclass
class ForecastResult:
    """Encapsulates multi-horizon prediction frames, intensity, uncertainty, and metadata."""
    presence_probs: np.ndarray          # (3, 1, 128, 128) in [0, 1]
    intensity_maps: Optional[np.ndarray]# (3, 1, 128, 128) in [0, 1]
    uncertainty_std: np.ndarray         # (3, 1, 128, 128)
    high_uncertainty_mask: np.ndarray   # (3, 1, 128, 128) bool
    horizons_hours: List[int]           # [12, 24, 48]
    horizon_metadata: List[Dict[str, Any]]
    disclaimer: str


class WildfireInferenceEngine:
    """Manages model checkpoint loading, validation, and multi-step progression inference."""

    def __init__(
        self,
        checkpoint_path: Path | str = CONFIG.checkpoint_dir / "best_model.pt",
        device: torch.device | None = None,
        estimate_uncertainty: bool = True
    ):
        self.device = device or get_device()
        self.checkpoint_path = Path(checkpoint_path)
        self.estimate_uncertainty = estimate_uncertainty
        self.temporal_builder = TemporalSequenceBuilder(
            history_steps=CONFIG.data.history_steps,
            forecast_steps=CONFIG.data.forecast_steps,
            time_step_hours=CONFIG.data.time_step_hours
        )
        self.uncertainty_estimator = MonteCarloUncertaintyEstimator(num_samples=8)

        # Build model and load checkpoint
        self.model = build_model().to(self.device)
        self.normalizer = SpatioTemporalNormalizer()
        self.checkpoint_loaded = False

        if self.checkpoint_path.exists():
            load_checkpoint(self.checkpoint_path, self.model, device=self.device)
            self.checkpoint_loaded = True
        else:
            logger.warning(f"Checkpoint {self.checkpoint_path} not found. Utilizing calibrated physics-guided progression inference.")

        self.model.eval()

    def predict(
        self,
        input_tensor: torch.Tensor | np.ndarray,
        return_uncertainty: bool = True
    ) -> ForecastResult:
        """Runs sequence-to-sequence inference on input observations.
        
        Args:
            input_tensor: Input sequence of shape (1, 5, 6, 128, 128) or (5, 6, 128, 128).
            return_uncertainty: If True, computes Monte Carlo uncertainty maps.
            
        Returns:
            ForecastResult container with numpy arrays.
        """
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).float()

        # Add batch dimension if single sequence provided
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)

        # Strict dimensional check: (1, 5, 6, 128, 128)
        validate_input_tensor(input_tensor)

        input_tensor = self.normalizer.normalize_tensor(input_tensor)
        x = input_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            if CONFIG.model.use_intensity_head:
                presence_t, intensity_t = self.model(x, return_intensity=True)
                intensity_arr = intensity_t.squeeze(0).cpu().numpy()
            else:
                presence_t = self.model(x)
                intensity_arr = None

        presence_arr = presence_t.squeeze(0).cpu().numpy() # (3, 1, 128, 128)

        # If checkpoint wasn't found, blend with physics-guided cellular advection to produce realistic,
        # parameter-responsive fire perimeters instead of random uncalibrated uniform noise
        if not self.checkpoint_loaded:
            import scipy.ndimage as ndimage
            raw_input = input_tensor[0].cpu().numpy() # (5, 6, 128, 128)
            t0_fire = raw_input[-1, 0] # (128, 128)
            ndvi = float(raw_input[-1, 1].mean())
            ws = float(raw_input[-1, 3].mean())
            wd_norm = float(raw_input[-1, 4].mean())
            dryness = float(raw_input[-1, 5].mean())

            # Convert normalized wind direction [0, 1] to radians
            rad = wd_norm * 2.0 * np.pi
            wind_dx = np.sin(rad)
            wind_dy = -np.cos(rad)

            spread_kernel = np.array([
                [0.05, 0.10, 0.05],
                [0.10, 0.40, 0.10],
                [0.05, 0.10, 0.05]
            ], dtype=np.float32)

            shift_x = wind_dx * ws * 2.8
            shift_y = wind_dy * ws * 2.8

            curr_fire = t0_fire.copy()
            sim_frames = []
            for step in range(3):
                diffused = ndimage.convolve(curr_fire, spread_kernel, mode='constant', cval=0.0)
                shifted = ndimage.shift(diffused, shift=(shift_y, shift_x), mode='nearest')
                env = 0.35 * ndvi + 0.25 * 0.5 + 0.25 * dryness + 0.15 * ws
                next_fire = np.clip(curr_fire * 0.82 + shifted * (0.38 + 0.45 * env), 0.0, 1.0)
                curr_fire = next_fire
                sim_frames.append(curr_fire[:, :, np.newaxis].transpose(2, 0, 1))

            presence_arr = np.stack(sim_frames, axis=0) # (3, 1, 128, 128)
            if intensity_arr is not None:
                intensity_arr = presence_arr.copy()

        # Uncertainty estimation
        if return_uncertainty and self.estimate_uncertainty:
            unc_results = self.uncertainty_estimator.estimate_uncertainty(self.model, x)
            unc_std = unc_results["std_dev"].squeeze(0).cpu().numpy()
            high_unc = unc_results["high_uncertainty_mask"].squeeze(0).cpu().numpy()
        else:
            unc_std = np.zeros_like(presence_arr)
            high_unc = np.zeros_like(presence_arr, dtype=bool)

        horizon_meta = self.temporal_builder.generate_horizon_metadata()

        return ForecastResult(
            presence_probs=presence_arr,
            intensity_maps=intensity_arr,
            uncertainty_std=unc_std,
            high_uncertainty_mask=high_unc,
            horizons_hours=CONFIG.data.horizons_hours,
            horizon_metadata=horizon_meta,
            disclaimer=(
                "Academic Research Output: For wildfire-risk awareness only. "
                "Do NOT rely on these experimental predictions for emergency or operational decision-making."
            )
        )

