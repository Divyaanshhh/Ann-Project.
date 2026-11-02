"""Autoregressive feedback adapter for multi-channel spatiotemporal models.

Ensures that 1-channel future fire predictions are properly formatted into 6-channel
input states by combining them with environmental condition assumptions (NDVI, DEM, Weather).
"""

from typing import Optional, Dict, Any
import numpy as np
import torch


class AutoregressiveFeedbackAdapter:
    """Adapts single-channel future fire predictions into 6-channel model input representations.
    
    Channels:
        C1: Fire observation (updated with model prediction)
        C2: NDVI vegetation (assumed held constant or depleted by fire)
        C3: DEM terrain risk (static topographical layer)
        C4: Wind speed (assumed held constant or updated via forecast)
        C5: Wind direction (assumed held constant or updated via forecast)
        C6: Atmospheric dryness (assumed held constant or updated via forecast)
    """

    def __init__(self, burn_depletion_factor: float = 0.35):
        self.burn_depletion_factor = burn_depletion_factor

    def adapt_prediction_to_input_step(
        self,
        predicted_fire_prob: torch.Tensor,
        previous_6ch_step: torch.Tensor,
        future_weather_override: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Constructs a valid (B, 6, H, W) input frame for the next autoregressive step.
        
        Args:
            predicted_fire_prob: Model output of shape (B, 1, H, W).
            previous_6ch_step: Preceding 6-channel frame of shape (B, 6, H, W).
            future_weather_override: Optional assumed meteorological forecast values.
            
        Returns:
            Tensor of shape (B, 6, H, W).
        """
        # C1: Update fire observation with new prediction
        c1_fire = predicted_fire_prob

        # C2: NDVI - deplete slightly where new fire was predicted
        prev_ndvi = previous_6ch_step[:, 1:2]
        c2_ndvi = torch.clamp(prev_ndvi - self.burn_depletion_factor * c1_fire, 0.0, 1.0)

        # C3: Static DEM terrain risk (preserved from previous observation)
        c3_terrain = previous_6ch_step[:, 2:3]

        # C4, C5, C6: Weather channels (either held constant or updated from override)
        if future_weather_override is not None:
            b, _, h, w = predicted_fire_prob.shape
            device = predicted_fire_prob.device

            ws_val = future_weather_override.get("wind_speed", previous_6ch_step[:, 3:4].mean().item())
            wd_val = future_weather_override.get("wind_direction", previous_6ch_step[:, 4:5].mean().item())
            dry_val = future_weather_override.get("dryness", previous_6ch_step[:, 5:6].mean().item())

            c4_ws = torch.full((b, 1, h, w), ws_val, device=device)
            c5_wd = torch.full((b, 1, h, w), wd_val, device=device)
            c6_dry = torch.full((b, 1, h, w), dry_val, device=device)
        else:
            c4_ws = previous_6ch_step[:, 3:4]
            c5_wd = previous_6ch_step[:, 4:5]
            c6_dry = previous_6ch_step[:, 5:6]

        adapted_step = torch.cat([c1_fire, c2_ndvi, c3_terrain, c4_ws, c5_wd, c6_dry], dim=1)
        return adapted_step

