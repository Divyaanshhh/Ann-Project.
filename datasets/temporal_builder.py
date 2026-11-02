"""Temporal sequence builder and sliding-window frame synchronizer."""

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import numpy as np


class TemporalSequenceBuilder:
    """Constructs non-leaking historical input sequences and multi-step forecast targets."""

    def __init__(
        self,
        history_steps: int = 5,
        forecast_steps: int = 3,
        time_step_hours: int = 12
    ):
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.time_step_hours = time_step_hours

    def generate_horizon_metadata(self, base_timestamp: datetime | None = None) -> List[Dict[str, Any]]:
        """Returns metadata for forecast frames with real-world hour offsets."""
        if base_timestamp is None:
            base_timestamp = datetime(2026, 8, 15, 12, 0, 0)

        horizons = []
        for step in range(1, self.forecast_steps + 1):
            offset_hours = step * self.time_step_hours
            target_time = base_timestamp + timedelta(hours=offset_hours)
            horizons.append({
                "frame_index": step - 1,
                "step_offset": step,
                "horizon_hours": offset_hours,
                "timestamp": target_time.isoformat(),
                "label": f"+{offset_hours} Hours Forecast"
            })
        return horizons

    def build_sliding_windows(
        self,
        continuous_series: np.ndarray,
        fire_channel_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts a continuous time-series (Total_Time, Channels, Height, Width) into (N, 5, C, H, W) and (N, 3, 1, H, W).
        
        Prevents temporal leakage by strictly partitioning past observations from future targets.
        """
        total_time, channels, h, w = continuous_series.shape
        window_size = self.history_steps + self.forecast_steps

        if total_time < window_size:
            raise ValueError(
                f"Continuous series length {total_time} is shorter than required window {window_size}"
            )

        inputs, targets = [], []
        for i in range(total_time - window_size + 1):
            x_win = continuous_series[i : i + self.history_steps]
            # Target is the fire channel only (or all channels if configured)
            y_win = continuous_series[
                i + self.history_steps : i + window_size,
                fire_channel_idx : fire_channel_idx + 1
            ]
            inputs.append(x_win)
            targets.append(y_win)

        return np.stack(inputs, axis=0), np.stack(targets, axis=0)

