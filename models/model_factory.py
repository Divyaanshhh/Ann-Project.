"""Factory for instantiating deep learning architectures (ConvLSTM and SpatioTemporal UNet)."""

import torch.nn as nn
from config import CONFIG
from models.convlstm_model import ConvLSTMForecaster
from models.spatiotemporal_unet import SpatioTemporalUNet
from utils.logger import get_logger

logger = get_logger("model_factory")


def build_model(
    model_type: str = CONFIG.model.model_type,
    in_channels: int = CONFIG.model.in_channels,
    hidden_dims: list = CONFIG.model.hidden_dims,
    out_channels: int = CONFIG.model.out_channels,
    pred_len: int = CONFIG.data.forecast_steps,
    seq_len: int = CONFIG.data.history_steps,
    dropout: float = CONFIG.model.dropout,
    use_intensity_head: bool = CONFIG.model.use_intensity_head
) -> nn.Module:
    """Builds and returns the configured neural network architecture.
    
    Args:
        model_type: 'convlstm' or 'spatiotemporal_unet'
        in_channels: Number of input channels (6).
        hidden_dims: Hidden dimensions list for ConvLSTM.
        out_channels: Number of output channels (1).
        pred_len: Prediction horizon frames (3).
        seq_len: Historical time steps (5).
        dropout: Dropout rate.
        use_intensity_head: Whether to instantiate secondary intensity head.
        
    Returns:
        Instantiated nn.Module.
    """
    m_type = model_type.lower().strip()
    if m_type == "convlstm":
        logger.info(f"Instantiating ConvLSTMForecaster with hidden_dims={hidden_dims}, pred_len={pred_len}")
        return ConvLSTMForecaster(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            out_channels=out_channels,
            kernel_size=CONFIG.model.kernel_size,
            pred_len=pred_len,
            dropout=dropout,
            use_intensity_head=use_intensity_head
        )
    elif m_type in ("spatiotemporal_unet", "unet"):
        logger.info(f"Instantiating SpatioTemporalUNet with in_channels={in_channels}, pred_len={pred_len}")
        return SpatioTemporalUNet(
            in_channels=in_channels,
            seq_len=seq_len,
            pred_len=pred_len,
            out_channels=out_channels,
            use_intensity_head=use_intensity_head
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'convlstm' or 'spatiotemporal_unet'.")
