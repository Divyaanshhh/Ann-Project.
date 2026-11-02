"""Logging infrastructure for FIRECAST AI."""

import logging
import sys
from pathlib import Path


def get_logger(name: str = "firecast", log_file: Path | None = None) -> logging.Logger:
    """Configures and returns a structured logger with console and optional file output.
    
    Args:
        name: Name of the logger component.
        log_file: Optional path to save log file.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

