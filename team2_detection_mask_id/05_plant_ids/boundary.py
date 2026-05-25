from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from scipy import ndimage
from shapely.geometry import Polygon, box, mapping, shape


def _resolve_reference_raster_path() -> Path:
    """Resolve the orthomosaic path used as the CRS reference for boundary reprojection."""
    env_path = os.environ.get("PLANT_ID_SOURCE_TIF")
    if env_path:
        candidate = Path(env_path).resolve()
        if candidate.exists():
            return candidate

    project_root = Path(__file__).resolve().parent.parent
    default_candidate = project_root / "ortho" / "ggs-orthophoto.tif"
    if default_candidate.exists():
        return default_candidate

    ortho_dir = project_root / "ortho"
    tif_candidates = sorted(ortho_dir.glob("*.tif")) + sorted(ortho_dir.glob("*.tiff"))
    if tif_candidates:
        return tif_candidates[0].resolve()

    raise FileNotFoundError(
        "Unable to resolve a reference orthomosaic for boundary reprojection. "
        "Set PLANT_ID_SOURCE_TIF or place a GeoTIFF inside the ortho folder."
    )


def load_boundary_from_geojson(geojson_path: str) -> Polygon:
    """Load a farm boundary GeoJSON and return a polygon in the orthomosaic CRS."""
    boundary_path = Path(geojson_path).resolve()
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary GeoJSON not found: {boundary_path}")

    reference_raster = _resolve_reference_raster_path()
    try:
        with rasterio.open(reference_raster) as src:
            source_crs = src.crs
    except Exception as exc:
        raise RuntimeError(f"Failed to open reference orthomosaic for CRS lookup: {reference_raster}") from exc

    try:
        boundary_gdf = gpd.read_file(boundary_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read boundary GeoJSON: {boundary_path}") from exc

    if boundary_gdf.empty:
        raise ValueError(f"Boundary GeoJSON is empty: {boundary_path}")
    if boundary_gdf.crs is None:
        raise ValueError(
            f"Boundary GeoJSON has no CRS defined: {boundary_path}. "
            "Provide a GeoJSON with a valid CRS."
        )

    if source_crs is None:
        raise ValueError(f"Reference orthomosaic has no CRS defined: {reference_raster}")

    if boundary_gdf.crs != source_crs:
        boundary_gdf = boundary_gdf.to_crs(source_crs)

    unified_geometry = boundary_gdf.geometry.unary_union
    if unified_geometry.is_empty:
        raise ValueError(f"Boundary geometry is empty after reprojection: {boundary_path}")

    polygon = unified_geometry.buffer(0)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    if not isinstance(polygon, Polygon):
        raise ValueError(f"Boundary geometry is not polygonal: {polygon.geom_type}")
    return polygon


def auto_detect_boundary_ndvi(source_tif_path: str, threshold: float = 0.15) -> Polygon:
    """Auto-detect the farm boundary from NDVI by extracting the largest vegetation polygon."""
    source_path = Path(source_tif_path).resolve()
    try:
        with rasterio.open(source_path) as src:
            if src.count < 2:
                raise ValueError(
                    f"Raster must contain at least two bands for NDVI boundary detection: {source_path}"
                )
            bands = src.read([1, 2]).astype(np.float32)
            transform = src.transform
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise RuntimeError(f"Failed to open raster for NDVI boundary detection: {source_path}") from exc

    red = bands[0]
    nir = bands[1]
    ndvi = (nir - red) / (nir + red + 1e-6)
    vegetation_mask = ndvi > threshold
    closed_mask = ndimage.binary_closing(vegetation_mask, iterations=2)
    if not closed_mask.any():
        raise ValueError(
            f"Auto boundary detection found no vegetation above NDVI threshold {threshold} in {source_path}"
        )

    polygon_candidates: list[Polygon] = []
    for geom, value in shapes(closed_mask.astype(np.uint8), mask=closed_mask.astype(bool), transform=transform):
        if int(value) != 1:
            continue
        candidate = shape(geom).buffer(0)
        if candidate.is_empty:
            continue
        if candidate.geom_type == "MultiPolygon":
            polygon_candidates.extend(candidate.geoms)
        elif isinstance(candidate, Polygon):
            polygon_candidates.append(candidate)

    if not polygon_candidates:
        raise ValueError(f"Failed to extract a polygon from the NDVI vegetation mask for {source_path}")

    return max(polygon_candidates, key=lambda geom: geom.area)


def clip_raster_to_boundary(source_tif_path: str, boundary_polygon: Polygon, output_path: str) -> None:
    """Clip a raster to the supplied boundary polygon and write the result preserving georeferencing."""
    source_path = Path(source_tif_path).resolve()
    destination_path = Path(output_path).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(source_path) as src:
            raster_bounds = box(*src.bounds)
            if not boundary_polygon.intersects(raster_bounds):
                raise ValueError(
                    f"Boundary polygon does not intersect raster extent for source {source_path}"
                )

            clipped_image, clipped_transform = mask(
                src,
                [mapping(boundary_polygon)],
                crop=True,
                filled=True,
            )
            output_meta = src.meta.copy()
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise RuntimeError(f"Failed to open or clip raster: {source_path}") from exc

    output_meta.update(
        {
            "height": clipped_image.shape[1],
            "width": clipped_image.shape[2],
            "transform": clipped_transform,
        }
    )

    try:
        with rasterio.open(destination_path, "w", **output_meta) as dst:
            dst.write(clipped_image)
    except Exception as exc:
        raise RuntimeError(f"Failed to write clipped raster: {destination_path}") from exc
