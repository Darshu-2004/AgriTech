from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol


def _read_two_band_ndvi_source(source_tif_path: str) -> np.ndarray:
    """Read the first two raster bands as float32 for NDVI-based processing."""
    source_path = Path(source_tif_path).resolve()
    try:
        with rasterio.open(source_path) as src:
            if src.count < 2:
                raise ValueError(f"Raster must contain at least two bands for NDVI processing: {source_path}")
            return src.read([1, 2]).astype(np.float32)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise RuntimeError(f"Failed to open raster for NDVI exclusion mask: {source_path}") from exc


def _to_uint8(rgb_data: np.ndarray) -> np.ndarray:
    """Normalize RGB raster data into uint8 for brightness-based masking."""
    if rgb_data.dtype == np.uint8:
        return rgb_data

    if np.issubdtype(rgb_data.dtype, np.integer):
        dtype_max = np.iinfo(rgb_data.dtype).max
        scaled = np.clip(rgb_data.astype(np.float32) / max(dtype_max, 1) * 255.0, 0, 255)
        return scaled.astype(np.uint8)

    finite = np.nan_to_num(rgb_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(finite.min())
    max_value = float(finite.max())
    if max_value <= min_value:
        return np.zeros_like(finite, dtype=np.uint8)
    scaled = (finite - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def build_ndvi_exclusion_mask(source_tif_path: str, low_threshold: float = 0.10) -> np.ndarray:
    """Build a boolean exclusion mask where low-NDVI pixels are marked as non-vegetation."""
    bands = _read_two_band_ndvi_source(source_tif_path)
    red = bands[0]
    nir = bands[1]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi <= low_threshold


def build_brightness_exclusion_mask(source_tif_path: str, high_threshold: float = 220) -> np.ndarray:
    """Build a boolean exclusion mask for very bright, likely non-plant surfaces."""
    source_path = Path(source_tif_path).resolve()
    try:
        with rasterio.open(source_path) as src:
            band_indexes = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
            rgb = src.read(band_indexes)
    except Exception as exc:
        raise RuntimeError(f"Failed to open raster for brightness exclusion mask: {source_path}") from exc

    rgb_uint8 = _to_uint8(np.transpose(rgb, (1, 2, 0)))
    brightness = rgb_uint8.mean(axis=2)
    return brightness > high_threshold


def merge_exclusion_masks(*masks: np.ndarray) -> np.ndarray:
    """Merge any number of boolean exclusion masks with a logical OR operation."""
    if not masks:
        raise ValueError("At least one exclusion mask is required")
    base_shape = masks[0].shape
    for index, mask in enumerate(masks, start=1):
        if mask.shape != base_shape:
            raise ValueError(f"Exclusion mask {index} has shape {mask.shape}, expected {base_shape}")
    return np.logical_or.reduce(masks)


def apply_exclusion_to_centroids(
    centroids: list[dict],
    exclusion_mask: np.ndarray,
    transform,
) -> list[dict]:
    """Filter out centroids that fall inside excluded raster regions."""
    filtered: list[dict] = []
    rows, cols = exclusion_mask.shape

    for centroid in centroids:
        if "geo_x" in centroid and "geo_y" in centroid:
            row, col = rowcol(transform, float(centroid["geo_x"]), float(centroid["geo_y"]))
        elif "pixel_x" in centroid and "pixel_y" in centroid:
            col = int(round(float(centroid["pixel_x"])))
            row = int(round(float(centroid["pixel_y"])))
        else:
            raise ValueError("Each centroid must include either geo_x/geo_y or pixel_x/pixel_y")

        if not (0 <= row < rows and 0 <= col < cols):
            continue
        if exclusion_mask[row, col]:
            continue

        updated = dict(centroid)
        updated["pixel_x"] = int(col)
        updated["pixel_y"] = int(row)
        filtered.append(updated)

    return filtered
