"""Model checkpoint saving and loading utilities for FIRECAST AI."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("checkpoint")


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    metrics: Dict[str, float],
    config_dict: Dict[str, Any],
    checkpoint_dir: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    is_best: bool = False,
    filename: Optional[str] = None
) -> Path:
    """Saves model weights, optimizer state, metrics, and metadata to disk.
    
    Args:
        model: PyTorch model instance.
        epoch: Current training epoch.
        metrics: Dictionary of metric scores (IoU, loss, Dice, etc.).
        config_dict: Serialized model and training configuration.
        checkpoint_dir: Destination directory.
        optimizer: Optional optimizer state.
        scheduler: Optional learning rate scheduler state.
        is_best: If True, also writes 'best_model.pt'.
        filename: Custom checkpoint filename.
        
    Returns:
        Path to saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": config_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }

    if filename is None:
        filename = f"checkpoint_epoch_{epoch:03d}.pt"
    
    ckpt_path = checkpoint_dir / filename
    torch.save(state, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)
        logger.info(f"Saved best model checkpoint to {best_path}")

    return ckpt_path


def load_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Tuple[int, Dict[str, float], Dict[str, Any]]:
    """Loads model weights, optimizer, and training state from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint .pt file.
        model: Model to populate with weights.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        device: Target torch device.
        
    Returns:
        Tuple of (epoch, metrics_dict, config_dict).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.save if False else torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model weights from {ckpt_path} (epoch {checkpoint.get('epoch', 0)})")

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return (
        checkpoint.get("epoch", 0),
        checkpoint.get("metrics", {}),
        checkpoint.get("config", {})
    )
