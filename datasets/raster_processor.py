"""Geospatial raster processing, alignment, and coordinate reference system (CRS) transformations."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
import torch

from utils.logger import get_logger

logger = get_logger("raster_processor")


class RasterProcessor:
    """Manages geospatial raster reading, reprojection, resampling, and tensor conversions."""

    def __init__(self, target_resolution: Tuple[int, int] = (128, 128), target_crs: str = "EPSG:4326"):
        self.target_height, self.target_width = target_resolution
        self.target_crs = target_crs

    def load_geotiff(self, filepath: Path | str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reads a GeoTIFF raster file with metadata preservation."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Raster file not found: {filepath}")

        with rasterio.open(filepath) as src:
            data = src.read(
                out_shape=(src.count, self.target_height, self.target_width),
                resampling=Resampling.bilinear
            )
            metadata = {
                "crs": str(src.crs),
                "transform": src.transform,
                "bounds": src.bounds,
                "count": src.count,
                "nodata": src.nodata,
                "width": src.width,
                "height": src.height,
            }

        # Handle NoData values
        if metadata["nodata"] is not None:
            data = np.where(data == metadata["nodata"], np.nan, data)

        return data, metadata

    def create_synthetic_geotiff(
        self,
        output_path: Path | str,
        data: np.ndarray,
        bounds: Tuple[float, float, float, float] = (-122.5, 37.0, -121.5, 38.0)
    ) -> Path:
        """Writes a multi-band NumPy array as an illustrative GeoTIFF raster with realistic CRS and transform."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        channels, height, width = data.shape
        west, south, east, north = bounds
        transform = from_bounds(west, south, east, north, width, height)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=channels,
            dtype=data.dtype,
            crs=self.target_crs,
            transform=transform,
            nodata=-9999.0
        ) as dst:
            dst.write(data)

        logger.info(f"Written GeoTIFF ({channels} bands, {height}x{width}) to {output_path}")
        return output_path

    @staticmethod
    def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
        """Safely converts numpy geospatial array to torch FloatTensor handling NaNs."""
        clean = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return torch.from_numpy(clean)

    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Converts PyTorch tensor back to detached NumPy array."""
        return tensor.detach().cpu().numpy()

