"""Interactive charts for training convergence, validation metrics, and ground truth comparisons."""

from typing import Dict, Any, List
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_training_history(history: Dict[str, List[float]]) -> go.Figure:
    """Creates a dual-panel interactive Plotly figure displaying loss and IoU/Dice curves."""
    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training & Validation Loss", "Validation Performance Metrics (IoU & Dice)"),
        horizontal_spacing=0.12
    )

    # Panel 1: Loss curves
    fig.add_trace(
        go.Scatter(x=epochs, y=history.get("train_loss", []), mode="lines+markers", name="Train Loss", line=dict(color="#F97316", width=2.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history.get("val_loss", []), mode="lines+markers", name="Val Loss", line=dict(color="#38BDF8", width=2.5, dash="dash")),
        row=1, col=1
    )

    # Panel 2: Metric curves
    fig.add_trace(
        go.Scatter(x=epochs, y=history.get("val_iou", []), mode="lines+markers", name="Val IoU", line=dict(color="#10B981", width=2.5)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history.get("val_dice", []), mode="lines+markers", name="Val Dice", line=dict(color="#A855F7", width=2.5)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history.get("val_f1", []), mode="lines", name="Val F1", line=dict(color="#FBBF24", width=1.5, dash="dot")),
        row=1, col=2
    )

    fig.update_layout(
        paper_bgcolor="#111318",
        plot_bgcolor="#1E222D",
        font=dict(color="#CBD5E1"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#334155"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    fig.update_xaxes(showgrid=True, gridcolor="#334155", title_text="Epoch")
    fig.update_yaxes(showgrid=True, gridcolor="#334155")

    return fig


def plot_confusion_overlay(prediction: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5) -> go.Figure:
    """Visualizes True Positives, False Positives, and False Negatives as an interactive segmented map."""
    pred_bin = (prediction >= threshold).astype(int)
    gt_bin = (ground_truth > 0.2).astype(int)

    # Categories:
    # 0 = Background
    # 1 = True Positive (Predicted fire & Actual fire) -> Green
    # 2 = False Positive (Predicted fire & No fire)   -> Amber/Orange
    # 3 = False Negative (Missed fire)                 -> Red
    overlay = np.zeros_like(pred_bin)
    overlay[(pred_bin == 1) & (gt_bin == 1)] = 1
    overlay[(pred_bin == 1) & (gt_bin == 0)] = 2
    overlay[(pred_bin == 0) & (gt_bin == 1)] = 3

    colorscale = [
        [0.0, "#1E222D"],      # Background
        [0.33, "#10B981"],     # True Positive
        [0.66, "#F59E0B"],     # False Positive
        [1.0, "#EF4444"]       # False Negative
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=overlay,
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            showscale=False
        )
    )

    fig.update_layout(
        title="Prediction vs Ground Truth Error Analysis (TP: Green | FP: Orange | FN: Red)",
        paper_bgcolor="#111318",
        plot_bgcolor="#1E222D",
        font=dict(color="#CBD5E1"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=360,
        margin=dict(l=30, r=30, t=40, b=30)
    )
    return fig
