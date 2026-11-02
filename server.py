"""FastAPI server powering the FIRECAST AI interactive geospatial website.

Exposes REST APIs for:
- Live GPU-accelerated inference (+12h, +24h, +48h)
- Epistemic uncertainty analysis (Monte Carlo Dropout)
- Real NASA FIRMS active fire feed
- Model evaluation metrics and training history
- Dynamic environmental simulation controls
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import CONFIG
from utils.device import get_device, get_device_info, clear_gpu_cache
from utils.checkpoint import load_checkpoint
from datasets.synthetic_generator import SyntheticWildfireGenerator
from datasets.real_wildfire_adapter import RealWildfireAdapter
from datasets.normalization import SpatioTemporalNormalizer
from models.model_factory import build_model
from forecasting.inference_engine import WildfireInferenceEngine
from training.metrics import compute_metrics

app = FastAPI(
    title="FIRECAST AI — Web API",
    description="Geospatial Wildfire Progression & Intensity Forecasting REST API",
    version="1.0.0"
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = CONFIG.project_root / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# Global cached engine
_engine: Optional[WildfireInferenceEngine] = None
_normalizer = SpatioTemporalNormalizer()


def get_engine() -> WildfireInferenceEngine:
    global _engine
    if _engine is None:
        ckpt = CONFIG.checkpoint_dir / "best_model.pt"
        _engine = WildfireInferenceEngine(checkpoint_path=ckpt)
    return _engine


class SimulationParams(BaseModel):
    wind_speed: float = 0.5           # [0, 1]
    wind_direction_deg: float = 45.0  # [0, 360]
    dryness: float = 0.8              # [0, 1]
    fuel_density: float = 0.7         # [0, 1]
    ignition_center_x: int = 64       # [0, 127]
    ignition_center_y: int = 64       # [0, 127]
    ignition_radius: int = 5          # [2, 15]


@app.get("/api/health")
def health():
    dev_info = get_device_info()
    return {
        "status": "online",
        "service": "FIRECAST AI",
        "device": dev_info
    }


@app.get("/api/metrics")
def get_metrics():
    """Returns training history and evaluation benchmark metrics."""
    history_file = CONFIG.checkpoint_dir / "training_history.json"
    eval_file = CONFIG.checkpoint_dir / "evaluation_results.json"

    history = {}
    evaluation = {}

    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)

    if eval_file.exists():
        with open(eval_file, "r") as f:
            evaluation = json.load(f)

    return {
        "history": history,
        "evaluation": evaluation
    }


@app.get("/api/dataset/sample")
def get_dataset_sample(source: str = "synthetic"):
    """Returns a full 6-channel spatiotemporal observation sequence and target."""
    if source == "firms":
        adapter = RealWildfireAdapter()
        df = adapter.fetch_or_load_real_firms_data()
        inputs, targets = adapter.rasterize_firms_cluster(df)
        label = "NASA FIRMS Satellite Active Fire (Real MODIS/VIIRS)"
    else:
        npz_path = CONFIG.synthetic_dir / "synthetic_wildfire_dataset.npz"
        if not npz_path.exists():
            gen = SyntheticWildfireGenerator(num_samples=CONFIG.data.num_synthetic_samples)
            gen.save_dataset(npz_path)
        data = np.load(npz_path)
        inputs = data["inputs"][0]
        targets = data["targets"][0]
        label = "Synthetic Wildfire Benchmark (128x128)"

    # Subsample or return rasters as nested lists
    # inputs: (5, 6, 128, 128), targets: (3, 1, 128, 128)
    return {
        "label": label,
        "history_steps": 5,
        "forecast_steps": 3,
        "channels": CONFIG.channels.names,
        # History fire channel (5, 128, 128)
        "history_fire": inputs[:, 0].tolist(),
        # Latest 6 channels at t0
        "current_channels": {
            "fire": inputs[-1, 0].tolist(),
            "ndvi": inputs[-1, 1].tolist(),
            "terrain": inputs[-1, 2].tolist(),
            "wind_speed": float(inputs[-1, 3].mean()),
            "wind_direction": float(inputs[-1, 4].mean() * 360.0),
            "dryness": float(inputs[-1, 5].mean())
        },
        "target_fire": targets[:, 0].tolist()
    }


@app.post("/api/predict")
def run_prediction(params: Optional[SimulationParams] = None):
    """Executes the deep learning model to forecast +12h, +24h, and +48h fire progression."""
    engine = get_engine()

    # If params are provided, dynamically construct customized 6-channel input
    if params is not None:
        h, w = CONFIG.data.spatial_height, CONFIG.data.spatial_width
        y, x = np.ogrid[:h, :w]
        dist_sq = (x - params.ignition_center_x) ** 2 + (y - params.ignition_center_y) ** 2
        fire_t0 = np.exp(-dist_sq / (2.0 * (params.ignition_radius ** 2))).astype(np.float32)

        # Build 5 history steps
        history = []
        for step in range(5):
            factor = 0.4 + 0.15 * step
            step_fire = np.clip(fire_t0 * factor, 0.0, 1.0)
            ndvi = np.full((h, w), params.fuel_density, dtype=np.float32)
            terrain = np.clip(0.3 + 0.4 * np.sin(x / 15.0) * np.cos(y / 15.0), 0.1, 0.9).astype(np.float32)
            ws = np.full((h, w), params.wind_speed, dtype=np.float32)
            wd = np.full((h, w), (params.wind_direction_deg % 360.0) / 360.0, dtype=np.float32)
            dry = np.full((h, w), params.dryness, dtype=np.float32)

            ch6 = np.stack([step_fire, ndvi, terrain, ws, wd, dry], axis=0)
            history.append(ch6)

        input_arr = np.stack(history, axis=0) # (5, 6, 128, 128)
    else:
        # Default sample
        npz_path = CONFIG.synthetic_dir / "synthetic_wildfire_dataset.npz"
        data = np.load(npz_path)
        input_arr = data["inputs"][0]

    # Run inference on GPU
    result = engine.predict(input_arr, return_uncertainty=True)

    # Compute statistics per frame
    horizon_stats = []
    for meta in result.horizon_metadata:
        idx = meta["frame_index"]
        p_frame = result.presence_probs[idx, 0]
        u_frame = result.uncertainty_std[idx, 0]
        active_pixels = int((p_frame >= 0.5).sum())
        max_prob = float(p_frame.max())
        mean_prob = float(p_frame.mean())
        mean_unc = float(u_frame.mean())
        max_unc = float(u_frame.max())

        horizon_stats.append({
            "horizon_hours": meta["horizon_hours"],
            "label": meta["label"],
            "active_pixels": active_pixels,
            "max_probability": round(max_prob, 4),
            "mean_probability": round(mean_prob, 4),
            "mean_uncertainty": round(mean_unc, 4),
            "max_uncertainty": round(max_unc, 4)
        })

    return {
        "status": "success",
        "horizons": [12, 24, 48],
        "stats": horizon_stats,
        "presence_probs": result.presence_probs[:, 0].tolist(), # (3, 128, 128)
        "uncertainty_std": result.uncertainty_std[:, 0].tolist(), # (3, 128, 128)
        "high_uncertainty_mask": result.high_uncertainty_mask[:, 0].tolist(),
        "disclaimer": result.disclaimer
    }


@app.get("/api/real-firms")
def get_real_firms():
    """Returns real NASA FIRMS satellite fire records."""
    adapter = RealWildfireAdapter()
    df = adapter.fetch_or_load_real_firms_data()
    top_records = df.head(100).to_dict(orient="records")
    return {
        "count": len(df),
        "records": top_records
    }


# Serve static assets and web interface
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)
