"""Centralized configuration for FIRECAST AI: Deep Wildfire Progression & Intensity Forecasting."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"


@dataclass
class ChannelConfig:
    """Six-channel multi-modal geospatial representation specifications."""
    num_channels: int = 6
    names: List[str] = field(default_factory=lambda: [
        "fire_history_intensity",      # C1: Historical fire perimeter & normalized intensity
        "ndvi_vegetation",             # C2: Normalized Difference Vegetation Index [0, 1]
        "terrain_risk_dem",            # C3: Slope & aspect combined terrain factor [0, 1]
        "wind_speed",                  # C4: Normalized wind speed [0, 1]
        "wind_direction",              # C5: Circular normalized wind angle [0, 1]
        "atmospheric_dryness",         # C6: Composite RH and temperature dryness index [0, 1]
    ])
    descriptions: List[str] = field(default_factory=lambda: [
        "Active fire mask and radiative intensity (C1)",
        "Sentinel-2 derived NDVI fuel density (C2)",
        "DEM slope and aspect fire spread coefficient (C3)",
        "Surface wind velocity magnitude (C4)",
        "Cyclic trigonometric wind direction (C5)",
        "Vapor pressure deficit & temperature dryness (C6)"
    ])


@dataclass
class DataConfig:
    """Spatiotemporal tensor dimensionality and splitting configuration."""
    batch_size: int = 8
    history_steps: int = 5          # Number of input time steps (T_in)
    forecast_steps: int = 3         # Number of future prediction steps (T_out)
    input_channels: int = 6         # 6 multi-modal channels
    target_channels: int = 1        # Binary fire presence mask (and optional intensity)
    spatial_height: int = 128       # Spatial grid height (pixels)
    spatial_width: int = 128        # Spatial grid width (pixels)
    
    # Train / Val / Test splits (temporal or spatial independence)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Real-world temporal horizon mapping
    # Step interval corresponds to 12-hour intervals for 12h, 24h, 48h targets (or 4h intervals)
    time_step_hours: int = 12
    horizons_hours: List[int] = field(default_factory=lambda: [12, 24, 48])
    
    # Synthetic dataset generator settings
    num_synthetic_samples: int = 100


@dataclass
class ModelConfig:
    """Deep learning model hyperparameters for ConvLSTM & SpatioTemporal UNet."""
    model_type: str = "convlstm"    # 'convlstm' or 'spatiotemporal_unet'
    in_channels: int = 6
    out_channels: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [32, 64])
    kernel_size: int = 3
    num_layers: int = 2
    dropout: float = 0.15
    use_intensity_head: bool = True  # Dual-head for presence + intensity


@dataclass
class TrainingConfig:
    """Training optimization and loss weights configuration."""
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 6
    
    # Custom Compound Loss parameters
    # Total Loss = focal_weight * Focal_Loss + intensity_weight * Weighted_MSE
    focal_weight: float = 1.0
    intensity_weight: float = 0.5
    focal_alpha: float = 0.75       # Emphasis on sparse positive fire pixels
    focal_gamma: float = 2.0        # Focusing factor for hard examples
    
    # Learning rate scheduler
    scheduler_type: str = "cosine"  # 'cosine', 'plateau', or 'step'
    lr_min: float = 1e-6
    
    # Metric evaluation threshold
    fire_threshold: float = 0.5


@dataclass
class FirecastConfig:
    """Master application configuration container."""
    project_name: str = "FIRECAST AI"
    version: str = "1.0.0"
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DATA_DIR
    processed_dir: Path = PROCESSED_DATA_DIR
    synthetic_dir: Path = SYNTHETIC_DATA_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    dataset_path: Path = PROCESSED_DATA_DIR / "real_wildfire_dataset.npz"
    
    def ensure_directories(self) -> None:
        """Create required project directories if they do not exist."""
        for d in [self.raw_dir, self.processed_dir, self.synthetic_dir, self.checkpoint_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Default global configuration instance
CONFIG = FirecastConfig()

