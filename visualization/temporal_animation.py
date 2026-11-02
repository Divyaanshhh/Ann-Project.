"""Interactive temporal animation and slider visualizer for past observations and future forecasts."""

from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go


def create_progression_animation(
    history_frames: np.ndarray,      # (5, 128, 128)
    forecast_frames: np.ndarray,     # (3, 128, 128)
    threshold: float = 0.5
) -> go.Figure:
    """Builds an interactive 8-step Plotly heat map animation connecting past observations with forecasts.
    
    Frames 1-5: Historical fire observations (t-4 to t0)
    Frames 6-8: Model-generated forecasts (+12h, +24h, +48h)
    """
    # Combine into 8 frames
    all_frames = np.concatenate([history_frames, forecast_frames], axis=0) # (8, 128, 128)
    frame_labels = [
        "Historical Frame t - 48h",
        "Historical Frame t - 36h",
        "Historical Frame t - 24h",
        "Historical Frame t - 12h",
        "Historical Frame t₀ (Current)",
        "Predicted +12h Forecast",
        "Predicted +24h Forecast",
        "Predicted +48h Forecast"
    ]
    frame_types = ["Observed"] * 5 + ["Forecast"] * 3

    # Initial frame
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=all_frames[0],
                colorscale="Inferno",
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="Fire Probability", tickfont=dict(color="#CBD5E1"), title_font=dict(color="#CBD5E1"))
            )
        ]
    )

    # Build slider frames
    plotly_frames = []
    for i in range(len(all_frames)):
        ftype_color = "#38BDF8" if frame_types[i] == "Observed" else "#F97316"
        title_text = f"<b>{frame_labels[i]}</b> &nbsp; <span style='color:{ftype_color}'>[{frame_types[i].upper()}]</span>"
        
        plotly_frames.append(
            go.Frame(
                data=[go.Heatmap(z=all_frames[i], colorscale="Inferno", zmin=0.0, zmax=1.0)],
                name=f"frame_{i}",
                layout=go.Layout(
                    title=dict(text=title_text, font=dict(color="#F8FAFC", size=15))
                )
            )
        )

    fig.frames = plotly_frames

    # Add slider & play/pause buttons
    fig.update_layout(
        title=dict(text=f"<b>{frame_labels[0]}</b> &nbsp; <span style='color:#38BDF8'>[OBSERVED]</span>", font=dict(color="#F8FAFC", size=15)),
        paper_bgcolor="#111318",
        plot_bgcolor="#1E222D",
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#64748B")),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#64748B"), autorange="reversed"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                direction="left",
                x=0.05,
                y=-0.15,
                pad=dict(r=10, t=10),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=650, redraw=True), fromcurrent=True, mode="immediate")]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                y=-0.15,
                x=0.25,
                len=0.72,
                steps=[
                    dict(
                        method="animate",
                        args=[[f"frame_{k}"], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=0))],
                        label=f"{k+1}: {frame_types[k][:3]}"
                    )
                    for k in range(len(all_frames))
                ],
                tickcolor="#64748B",
                font=dict(color="#CBD5E1")
            )
        ]
    )

    return fig
