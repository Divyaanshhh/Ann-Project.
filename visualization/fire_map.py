"""Geospatial mapping utilities using Folium and Matplotlib for wildfire overlays."""

from typing import Tuple, Optional
import numpy as np
import folium
from folium import plugins
import matplotlib.cm as cm


def create_folium_fire_map(
    fire_mask: np.ndarray,
    bounds: Tuple[float, float, float, float] = (-121.5, 39.7, -121.3, 39.9), # California region (Camp Fire)
    center_lat: float = 39.8,
    center_lon: float = -121.4,
    zoom_start: int = 11,
    layer_name: str = "Active Fire Probability"
) -> folium.Map:
    """Renders interactive Folium map with raster fire probability heatmap overlay.
    
    Args:
        fire_mask: 2D array (H, W) in range [0, 1].
        bounds: (min_lon, min_lat, max_lon, max_lat).
        center_lat: Latitude of map center.
        center_lon: Longitude of map center.
        zoom_start: Initial zoom level.
        layer_name: Map layer label.
        
    Returns:
        Configured folium.Map instance.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB dark_matter"
    )

    min_lon, min_lat, max_lon, max_lat = bounds

    # Colorize raster mask with fire colormap (YlOrRd / magma)
    norm_fire = np.clip(fire_mask, 0.0, 1.0)
    
    # Generate RGBA image
    colormap = cm.get_cmap("inferno")
    rgba = colormap(norm_fire)
    # Mask out background below 0.15 probability (make transparent)
    rgba[..., 3] = np.where(norm_fire > 0.15, norm_fire * 0.85, 0.0)
    rgba_bytes = (rgba * 255).astype(np.uint8)

    # Convert to PNG in memory
    from PIL import Image
    import io
    import base64

    img = Image.fromarray(rgba_bytes, mode="RGBA")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Add ImageOverlay
    folium.raster_layers.ImageOverlay(
        image=data_uri,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.8,
        name=layer_name,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m
