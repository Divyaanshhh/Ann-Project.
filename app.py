"""FIRECAST AI — Deep Wildfire Progression & Intensity Forecasting System.

Interactive Geospatial Research Dashboard built with Streamlit, PyTorch, Plotly, and Folium.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from config import CONFIG
from utils.device import get_device, get_device_info
from datasets.data_loader import create_dataloaders
from datasets.synthetic_generator import SyntheticWildfireGenerator
from datasets.real_wildfire_adapter import RealWildfireAdapter
from forecasting.inference_engine import WildfireInferenceEngine
from visualization.fire_map import create_folium_fire_map
from visualization.geospatial_visualizer import plot_multipanel_forecast, plot_multimodal_channels_grid
from visualization.temporal_animation import create_progression_animation
from visualization.metrics_dashboard import plot_training_history, plot_confusion_overlay

# Set Streamlit page configuration with dark theme
st.set_page_config(
    page_title="FIRECAST AI — Wildfire Forecasting",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark scientific CSS styling
st.markdown("""
<style>
    .stApp {
        background-color: #0B0E14;
        color: #F8FAFC;
    }
    .metric-card {
        background: linear-gradient(135deg, #161B26 0%, #1F2737 100%);
        border: 1px solid #2D3748;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #F97316;
    }
    .metric-label {
        font-size: 13px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .disclaimer-box {
        background-color: #1E1B18;
        border-left: 4px solid #EAB308;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 13px;
        color: #FEF08A;
        margin: 15px 0;
    }
    .section-title {
        color: #F8FAFC;
        font-weight: 600;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 6px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_inference_engine():
    """Initializes and caches the deep learning inference engine."""
    ckpt_path = CONFIG.checkpoint_dir / "best_model.pt"
    return WildfireInferenceEngine(checkpoint_path=ckpt_path)


@st.cache_data
def load_sample_data(use_real_firms: bool = False):
    """Loads either real NASA FIRMS active fire raster or synthetic multi-modal sample."""
    if use_real_firms:
        adapter = RealWildfireAdapter()
        df = adapter.fetch_or_load_real_firms_data()
        x, y = adapter.rasterize_firms_cluster(df)
        return x, y, "Real NASA FIRMS Satellite Observations (MODIS / VIIRS)"
    else:
        npz_path = CONFIG.synthetic_dir / "synthetic_wildfire_dataset.npz"
        if not npz_path.exists():
            gen = SyntheticWildfireGenerator(num_samples=CONFIG.data.num_synthetic_samples)
            gen.save_dataset(npz_path)
        data = np.load(npz_path)
        inputs = data["inputs"]
        targets = data["targets"]
        return inputs[0], targets[0], "Synthetic Benchmark Spatiotemporal Dataset (100 Samples)"


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/fire-element.png", width=64)
    st.sidebar.title("FIRECAST AI")
    st.sidebar.caption("Deep Wildfire Progression & Intensity Forecasting")

    nav_option = st.sidebar.radio(
        "Navigation",
        [
            "Overview Dashboard",
            "Historical Fire Analysis",
            "Future Forecast Visualization",
            "Temporal Animation",
            "Model Performance & Metrics",
            "Uncertainty & Reliability",
            "Dataset & Model Management"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Source")
    data_source = st.sidebar.selectbox(
        "Select Data Stream",
        ["Synthetic Benchmark Dataset", "Real NASA FIRMS Satellite Feed"]
    )
    use_real = (data_source == "Real NASA FIRMS Satellite Feed")

    # Global threshold slider
    fire_threshold = st.sidebar.slider(
        "Fire Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability threshold defining active fire perimeter"
    )

    # Device Status in Sidebar
    dev_info = get_device_info()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"⚡ Device: **{dev_info['device_name']}**")
    if dev_info.get("cuda_available"):
        st.sidebar.caption(f"GPU VRAM: {dev_info.get('total_vram_gb')} GB (CUDA active)")

    # Load data and model
    input_sample, target_sample, data_source_label = load_sample_data(use_real_firms=use_real)
    engine = load_inference_engine()

    # Academic Safety Disclaimer Header
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <b>ACADEMIC RESEARCH & WILDFIRE-RISK-AWARENESS PROTOTYPE</b><br>
        Experimental predictions are produced by Deep Spatio-Temporal neural networks for research purposes.
        Never use this experimental model for emergency evacuation, tactical firefighting, or operational deployment.
        Direct all disaster management decisions to official authorities.
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # SECTION A: OVERVIEW DASHBOARD
    # ----------------------------------------------------
    if nav_option == "Overview Dashboard":
        st.title("🔥 FIRECAST AI — Geospatial Research Platform")
        st.markdown(
            "Predicting wildfire perimeter progression and radiative intensity using "
            "**Spatio-Temporal Deep Learning (ConvLSTM)** over 6 multi-modal geospatial channels."
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Model Architecture</div>
                <div class="metric-value">ConvLSTM</div>
                <span style="font-size:12px; color:#94A3B8;">Stacked Recurrent Conv [32, 64]</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Forecast Horizons</div>
                <div class="metric-value">+12h, +24h, +48h</div>
                <span style="font-size:12px; color:#94A3B8;">Multi-step Autoregressive</span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Input Representation</div>
                <div class="metric-value">6 Channels</div>
                <span style="font-size:12px; color:#94A3B8;">Fire, NDVI, DEM, Wind, Dryness</span>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Hardware Acceleration</div>
                <div class="metric-value">{"GPU Active" if dev_info['cuda_available'] else "CPU"}</div>
                <span style="font-size:12px; color:#94A3B8;">{dev_info['device_name']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("System Pipeline Overview")
        st.markdown("""
        ```
        Raw Multi-Modal Data           6-Channel Preprocessor          Stacked ConvLSTM Engine         Forecast & Uncertainty
        [NASA FIRMS / Sentinel-2]  -->  (Batch, 5, 6, 128, 128)   -->  Spatiotemporal Feature Map  -->  Presence [12h, 24h, 48h]
        [DEM Slope / Aspect]            Raster Alignment & Norm         Hidden State Propagation        Intensity Head + MC Dropout
        [Wind Vectors & Dryness]        Temporal Windowing              Spatial Convolutions            Interactive UI Dashboard
        ```
        """)

        st.info(f"Currently viewing active data stream: **{data_source_label}** (Grid: 128x128 pixels)")

    # ----------------------------------------------------
    # SECTION B: HISTORICAL FIRE ANALYSIS
    # ----------------------------------------------------
    elif nav_option == "Historical Fire Analysis":
        st.title("🛰️ Multi-Modal Historical Geospatial Layers")
        st.markdown("Inspect the 6 synchronized input channels driving the spatio-temporal model at current observation step ($t_0$).")

        # Plot 6-channel grid
        fig_channels = plot_multimodal_channels_grid(input_sample[-1], CONFIG.channels.names)
        st.pyplot(fig_channels)

        st.markdown("### Interactive Geographical Map (Current Fire Front)")
        m = create_folium_fire_map(input_sample[-1, 0], layer_name="Historical Fire Observation (t0)")
        st_folium(m, width=950, height=450)

    # ----------------------------------------------------
    # SECTION C: FUTURE FORECAST VISUALIZATION
    # ----------------------------------------------------
    elif nav_option == "Future Forecast Visualization":
        st.title("🔮 Multi-Horizon Wildfire Progression Forecast")
        st.markdown("Synchronized comparison across **Current Observation**, **+12h**, **+24h**, and **+48h** prediction horizons.")

        # Run inference
        result = engine.predict(input_sample, return_uncertainty=True)

        # Plot multi-panel forecast
        fig_forecast = plot_multipanel_forecast(
            historical_last_step=input_sample[-1, 0],
            forecast_presence=result.presence_probs,
            uncertainty_std=result.uncertainty_std,
            threshold=fire_threshold
        )
        st.pyplot(fig_forecast)

        # Display numeric summary table
        st.subheader("Forecast Summary Statistics")
        stats_data = []
        for meta in result.horizon_metadata:
            step_idx = meta["frame_index"]
            frame_prob = result.presence_probs[step_idx, 0]
            active_pixels = int((frame_prob >= fire_threshold).sum())
            max_intensity = float(frame_prob.max())
            mean_unc = float(result.uncertainty_std[step_idx, 0].mean())
            stats_data.append({
                "Horizon": meta["label"],
                "Active Fire Pixels": active_pixels,
                "Max Estimated Probability": f"{max_intensity:.3f}",
                "Mean Epistemic Uncertainty": f"{mean_unc:.4f}",
                "Observation Interval": f"{meta['step_offset'] * 12} Hours"
            })
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    # ----------------------------------------------------
    # SECTION D: TEMPORAL ANIMATION
    # ----------------------------------------------------
    elif nav_option == "Temporal Animation":
        st.title("⏱️ Spatio-Temporal Progression Animation")
        st.markdown("Play through 5 historical observation frames followed seamlessly by 3 autoregressively predicted horizons.")

        result = engine.predict(input_sample, return_uncertainty=False)
        history_fire = input_sample[:, 0]  # (5, 128, 128)
        forecast_fire = result.presence_probs[:, 0]  # (3, 128, 128)

        fig_anim = create_progression_animation(history_fire, forecast_fire, threshold=fire_threshold)
        st.plotly_chart(fig_anim, use_container_width=True)

    # ----------------------------------------------------
    # SECTION E: MODEL PERFORMANCE & METRICS
    # ----------------------------------------------------
    elif nav_option == "Model Performance & Metrics":
        st.title("📈 Model Training Convergence & Validation Metrics")

        history_file = CONFIG.checkpoint_dir / "training_history.json"
        eval_file = CONFIG.checkpoint_dir / "evaluation_results.json"

        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)
            fig_hist = plot_training_history(history)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Training history not found. Train the model using `python -m training.trainer`.")

        if eval_file.exists():
            with open(eval_file, "r") as f:
                eval_res = json.load(f)
            st.subheader("Test Split Evaluation Report")
            om = eval_res.get("overall_metrics", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test IoU", f"{om.get('iou', 0.0):.4f}")
            c2.metric("Test Dice", f"{om.get('dice', 0.0):.4f}")
            c3.metric("Test Precision", f"{om.get('precision', 0.0):.4f}")
            c4.metric("Test Recall", f"{om.get('recall', 0.0):.4f}")

        # Ground truth error analysis on target
        result = engine.predict(input_sample, return_uncertainty=False)
        st.subheader("Prediction vs Ground Truth Error Segmentation (+24h)")
        fig_err = plot_confusion_overlay(result.presence_probs[1, 0], target_sample[1, 0], threshold=fire_threshold)
        st.plotly_chart(fig_err, use_container_width=True)

    # ----------------------------------------------------
    # SECTION F: UNCERTAINTY & RELIABILITY
    # ----------------------------------------------------
    elif nav_option == "Uncertainty & Reliability":
        st.title("🛡️ Epistemic Uncertainty & Reliability Analysis")
        st.markdown("Monte Carlo Dropout generates empirical confidence bounds and flags high-variance perimeter boundaries.")

        result = engine.predict(input_sample, return_uncertainty=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted Mean Probability (+48h)")
            st.image(result.presence_probs[2, 0], clamp=True, use_container_width=True)
        with col2:
            st.subheader("Epistemic Uncertainty / Variance (+48h)")
            st.image(result.uncertainty_std[2, 0], clamp=True, use_container_width=True)

        st.warning(
            "High uncertainty typically concentrates along rapidly advancing frontal edges where wind turbulence "
            "and fuel micro-variations introduce physical stochasticity."
        )

    # ----------------------------------------------------
    # SECTION G: DATASET & MODEL MANAGEMENT
    # ----------------------------------------------------
    elif nav_option == "Dataset & Model Management":
        st.title("⚙️ Dataset & Model Checkpoint Management")

        st.subheader("Dataset Operations")
        if st.button("Regenerate Synthetic Dataset (100 Samples)"):
            with st.spinner("Generating physically motivated synthetic dataset..."):
                gen = SyntheticWildfireGenerator(num_samples=CONFIG.data.num_synthetic_samples)
                gen.save_dataset(CONFIG.synthetic_dir / "synthetic_wildfire_dataset.npz")
                st.success("Synthetic dataset successfully regenerated!")

        if st.button("Fetch Real NASA FIRMS Satellite Feed"):
            with st.spinner("Connecting to NASA FIRMS active fire feed..."):
                adapter = RealWildfireAdapter()
                df = adapter.fetch_or_load_real_firms_data()
                st.success(f"Acquired {len(df)} real fire observations!")
                st.dataframe(df.head(10))

        st.subheader("Saved Model Checkpoints")
        ckpts = list(CONFIG.checkpoint_dir.glob("*.pt"))
        if ckpts:
            for ckpt in ckpts:
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                st.write(f"📁 **{ckpt.name}** ({size_mb:.2f} MB)")
        else:
            st.write("No checkpoints found. Run training script to produce `best_model.pt`.")


if __name__ == "__main__":
    main()

