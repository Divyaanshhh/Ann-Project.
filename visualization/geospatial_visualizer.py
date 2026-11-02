"""Geospatial multi-panel visualizer for multi-horizon forecast comparisons."""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_multipanel_forecast(
    historical_last_step: np.ndarray,
    forecast_presence: np.ndarray,
    uncertainty_std: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    title: str = "FIRECAST AI — Spatiotemporal Wildfire Progression Forecast"
) -> plt.Figure:
    """Creates a synchronized 5-panel matplotlib figure comparing observation, horizons, and uncertainty.
    
    Args:
        historical_last_step: (128, 128) latest observed fire mask.
        forecast_presence: (3, 1, 128, 128) predictions for +12h, +24h, +48h.
        uncertainty_std: Optional (3, 1, 128, 128) epistemic standard deviation.
        threshold: Fire classification threshold.
        title: Figure title.
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), facecolor="#111318")
    fig.suptitle(title, color="#F8FAFC", fontsize=16, fontweight="bold", y=1.02)

    panels = [
        ("Historical (t₀)", historical_last_step, "inferno", [0.0, 1.0]),
        ("Forecast +12 Hours", forecast_presence[0, 0], "inferno", [0.0, 1.0]),
        ("Forecast +24 Hours", forecast_presence[1, 0], "inferno", [0.0, 1.0]),
        ("Forecast +48 Hours", forecast_presence[2, 0], "inferno", [0.0, 1.0]),
    ]

    for idx, (label, data, cmap, clim) in enumerate(panels):
        ax = axes[idx]
        ax.set_facecolor("#1E222D")
        im = ax.imshow(data, cmap=cmap, vmin=clim[0], vmax=clim[1], origin="upper")
        # Overlay threshold contour
        if (data >= threshold).any():
            ax.contour(data >= threshold, levels=[0.5], colors=["#FF4500"], linewidths=1.5)
        ax.set_title(label, color="#CBD5E1", fontsize=12, pad=8)
        ax.tick_params(colors="#64748B", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#CBD5E1", labelsize=8)

    # 5th panel: Uncertainty at +48h
    ax_unc = axes[4]
    ax_unc.set_facecolor("#1E222D")
    if uncertainty_std is not None:
        unc_data = uncertainty_std[2, 0] # 48h horizon uncertainty
        im_u = ax_unc.imshow(unc_data, cmap="viridis", origin="upper")
        ax_unc.set_title("Uncertainty (+48h StdDev)", color="#CBD5E1", fontsize=12, pad=8)
        cbar_u = fig.colorbar(im_u, ax=ax_unc, fraction=0.046, pad=0.04)
        cbar_u.ax.tick_params(colors="#CBD5E1", labelsize=8)
    else:
        ax_unc.text(0.5, 0.5, "Uncertainty N/A", color="#94A3B8", ha="center", va="center")
        ax_unc.set_title("Uncertainty", color="#CBD5E1", fontsize=12)

    ax_unc.tick_params(colors="#64748B", labelsize=8)
    for spine in ax_unc.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    return fig


def plot_multimodal_channels_grid(step_6ch: np.ndarray, channel_names: List[str]) -> plt.Figure:
    """Plots all 6 input channels of an observation frame in a 2x3 grid."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="#111318")
    fig.suptitle("Six-Channel Multi-Modal Geospatial Representation", color="#F8FAFC", fontsize=15, fontweight="bold")

    cmaps = ["inferno", "YlGn", "terrain", "Blues", "twilight", "Oranges"]

    for i in range(6):
        r, c = i // 3, i % 3
        ax = axes[r, c]
        ax.set_facecolor("#1E222D")
        im = ax.imshow(step_6ch[i], cmap=cmaps[i], origin="upper")
        name = channel_names[i] if i < len(channel_names) else f"Channel {i+1}"
        ax.set_title(name, color="#CBD5E1", fontsize=11, pad=6)
        ax.tick_params(colors="#64748B", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#CBD5E1", labelsize=8)

    plt.tight_layout()
    return fig
