"""Device management and GPU acceleration utilities for FIRECAST AI."""

import torch
from utils.logger import get_logger

logger = get_logger("device")


def get_device(force_cpu: bool = False) -> torch.device:
    """Detects and returns optimal PyTorch device (CUDA GPU or CPU).
    
    Args:
        force_cpu: If True, forces CPU usage even if CUDA GPU is present.
        
    Returns:
        torch.device instance.
    """
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Using GPU device: {gpu_name} ({gpu_memory:.2f} GB VRAM)")
        return device
    else:
        logger.info("Using CPU device.")
        return torch.device("cpu")


def get_device_info() -> dict:
    """Returns a dictionary containing hardware and device details."""
    cuda_available = torch.cuda.is_available()
    info = {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "current_device": str(get_device()),
    }
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        info["total_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["allocated_vram_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
        info["cached_vram_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)
    return info


def clear_gpu_cache() -> None:
    """Clears PyTorch CUDA memory cache if GPU is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

