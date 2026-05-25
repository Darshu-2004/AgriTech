from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import xy
from scipy import ndimage
from shapely.geometry import Point, mapping, shape

from boundary import auto_detect_boundary_ndvi, load_boundary_from_geojson
from clustering import assign_row_col_within_sector, cluster_plants_hdbscan, format_sector_label
from exclusion import (
    apply_exclusion_to_centroids,
    build_brightness_exclusion_mask,
    build_ndvi_exclusion_mask,
    merge_exclusion_masks,
)
from export import export_geojson, export_summary_json
from generate_viewer import generate_viewer
from plant_id import assign_plant_ids


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the plant ID post-processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate clustered plant IDs from an existing orthomosaic run summary and source raster."
    )
    parser.add_argument("--source", required=True, help="Path to the original orthomosaic GeoTIFF.")
    parser.add_argument("--run-summary", required=True, help="Path to the existing run_summary.json file.")
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Path to orthomosaic_mask.tif. If provided, centroids are extracted from the mask raster instead of "
        "run-summary JSON. Recommended when tile_summary in run_summary.json is empty.",
    )
    parser.add_argument("--boundary", help="Optional GeoJSON farm boundary. If omitted, NDVI auto-detection is used.")
    parser.add_argument(
        "--output-dir",
        default="pipeline_outputs/plant_ids",
        help="Directory for GeoJSON and plant ID summary outputs.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=25,
        help="HDBSCAN min_cluster_size value; default 25, increase for larger farms",
    )
    parser.add_argument("--ndvi-threshold", type=float, default=0.15, help="NDVI threshold for boundary detection.")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    """Load a JSON file into a Python dictionary."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON file: {path}") from exc


def load_centroids_from_mask(mask_path: Path) -> list[dict]:
    """
    Extract one centroid per connected plant region from the stitched
    mask raster. Uses rasterio.features.shapes to find connected blobs,
    then computes each blob's centroid and area in georeferenced coords.

    Returns a list of dicts with keys:
        geo_x       float  centroid easting in raster CRS
        geo_y       float  centroid northing in raster CRS
        pixel_x     int    centroid column in raster pixel space
        pixel_y     int    centroid row in raster pixel space
        area_px     int    blob area in pixels
        confidence  float  always 1.0 (no confidence from mask raster)
    """
    import numpy as np
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    centroids = []
    with rasterio.open(mask_path) as ds:
        transform = ds.transform
        data = ds.read(1)
        inv_transform = ~transform

        for geom_dict, val in shapes(data, mask=(data > 0), transform=transform):
            if val == 0:
                continue
            poly = shape(geom_dict)
            c = poly.centroid
            px, py = inv_transform * (c.x, c.y)
            centroids.append(
                {
                    "geo_x": c.x,
                    "geo_y": c.y,
                    "pixel_x": int(round(px)),
                    "pixel_y": int(round(py)),
                    "area_px": int(round(poly.area)),
                    "confidence": 1.0,
                }
            )

    print(f"Extracted {len(centroids)} plant centroids from mask raster.")
    return centroids


def load_centroids_from_summary(run_summary_path: Path) -> list[dict]:
    """Load centroid records from run_summary.json when explicit centroid objects are available."""
    run_summary = _load_json(run_summary_path)
    if isinstance(run_summary.get("centroids"), list):
        return [dict(item) for item in run_summary["centroids"]]

    tile_summary = run_summary.get("tile_summary")
    if tile_summary is None:
        tile_summary = run_summary.get("per_tile_summary")
    if isinstance(tile_summary, list):
        collected: list[dict] = []
        for tile in tile_summary:
            detections = tile.get("detections")
            if isinstance(detections, list):
                for detection in detections:
                    collected.append(dict(detection))
        if collected:
            return collected
    return []


def _ensure_geo_coordinates(centroids: list[dict], transform) -> list[dict]:
    """Ensure every centroid record contains both pixel and geographic coordinates."""
    normalized: list[dict] = []
    for centroid in centroids:
        updated = dict(centroid)
        if "geo_x" in updated and "geo_y" in updated:
            if "pixel_x" not in updated or "pixel_y" not in updated:
                row, col = rasterio.transform.rowcol(transform, float(updated["geo_x"]), float(updated["geo_y"]))
                updated["pixel_x"] = int(col)
                updated["pixel_y"] = int(row)
        elif "pixel_x" in updated and "pixel_y" in updated:
            geo_x, geo_y = xy(transform, float(updated["pixel_y"]), float(updated["pixel_x"]), offset="center")
            updated["geo_x"] = float(geo_x)
            updated["geo_y"] = float(geo_y)
        else:
            raise ValueError("Each centroid must contain either geo_x/geo_y or pixel_x/pixel_y")
        normalized.append(updated)
    return normalized


def _filter_centroids_to_boundary(centroids: list[dict], boundary_polygon) -> list[dict]:
    """Keep only centroid records that fall within the farm boundary polygon."""
    filtered: list[dict] = []
    for centroid in centroids:
        point = Point(float(centroid["geo_x"]), float(centroid["geo_y"]))
        if boundary_polygon.contains(point) or boundary_polygon.touches(point):
            filtered.append(dict(centroid))
    return filtered


def _print_summary_table(centroids: list[dict]) -> None:
    """Print a compact sector summary table to stdout."""
    sector_counts = Counter(format_sector_label(int(item["sector_id"])) for item in centroids if int(item["sector_id"]) != -1)
    noise_count = sum(1 for item in centroids if int(item["sector_id"]) == -1)

    print("")
    print("Plant ID Pipeline Summary")
    print("-------------------------")
    print(f"Total plants : {len(centroids)}")
    print(f"Noise plants : {noise_count}")
    print("Plants by sector:")
    if not sector_counts:
        print("  (none)")
        return
    for sector_label, count in sorted(sector_counts.items()):
        print(f"  {sector_label:<8} {count}")


def _write_boundary_geojson(boundary_polygon, output_path: Path) -> None:
    """Write the resolved farm boundary as a GeoJSON FeatureCollection."""
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"layer": "farm_boundary"},
                "geometry": mapping(boundary_polygon),
            }
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_exclusion_outline_geojson(exclusion_mask: np.ndarray, transform, output_path: Path) -> None:
    """Write exclusion-mask polygon outlines as a GeoJSON FeatureCollection."""
    features: list[dict] = []
    for geom, value in shapes(exclusion_mask.astype(np.uint8), mask=exclusion_mask.astype(bool), transform=transform):
        if int(value) != 1:
            continue
        polygon = shape(geom).buffer(0)
        if polygon.is_empty:
            continue
        if polygon.geom_type == "MultiPolygon":
            for part in polygon.geoms:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"layer": "exclusion_zone"},
                        "geometry": mapping(part),
                    }
                )
        else:
            features.append(
                {
                    "type": "Feature",
                    "properties": {"layer": "exclusion_zone"},
                    "geometry": mapping(polygon),
                }
            )

    payload = {"type": "FeatureCollection", "features": features}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_empty_geojson(output_path: Path, layer_name: str) -> None:
    """Write an empty GeoJSON layer placeholder."""
    payload = {"type": "FeatureCollection", "features": []}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Run the plant ID assignment pipeline from the command line."""
    args = parse_args()
    source_path = Path(args.source).resolve()
    run_summary_path = Path(args.run_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Source orthomosaic not found: {source_path}")
    if not run_summary_path.exists():
        raise FileNotFoundError(f"run_summary.json not found: {run_summary_path}")

    os.environ["PLANT_ID_SOURCE_TIF"] = str(source_path)

    run_summary = _load_json(run_summary_path)
    try:
        with rasterio.open(source_path) as src:
            transform = src.transform
            source_crs_wkt = src.crs.to_wkt() if src.crs else ""
            raster_bounds_polygon = shape(
                {
                    "type": "Polygon",
                    "coordinates": [[
                        [src.bounds.left, src.bounds.bottom],
                        [src.bounds.left, src.bounds.top],
                        [src.bounds.right, src.bounds.top],
                        [src.bounds.right, src.bounds.bottom],
                        [src.bounds.left, src.bounds.bottom],
                    ]],
                }
            )
    except Exception as exc:
        raise RuntimeError(f"Failed to open source orthomosaic: {source_path}") from exc

    if args.mask:
        centroids = load_centroids_from_mask(args.mask)
    else:
        centroids = load_centroids_from_summary(run_summary_path)
        if len(centroids) == 0:
            print("WARNING: No centroids found in run_summary.json.")
            print("Tip: pass --mask path/to/orthomosaic_mask.tif instead.")
            sys.exit(1)

    centroids = _ensure_geo_coordinates(centroids, transform)

    if args.boundary:
        boundary_polygon = load_boundary_from_geojson(args.boundary)
        boundary_mode = "manual"
        centroids = _filter_centroids_to_boundary(centroids, boundary_polygon)
    else:
        boundary_polygon = raster_bounds_polygon
        boundary_mode = "raster_extent"

    centroids = cluster_plants_hdbscan(centroids, min_cluster_size=args.min_cluster_size)
    centroids = assign_row_col_within_sector(centroids)
    centroids = assign_plant_ids(centroids)

    geojson_path = output_dir / "plants.geojson"
    summary_path = output_dir / "plant_id_summary.json"
    boundary_path = output_dir / "farm_boundary.geojson"
    exclusion_outline_path = output_dir / "exclusion_mask_outline.geojson"
    viewer_path = output_dir / "viewer.html"

    export_geojson(centroids, str(geojson_path), source_crs_wkt)
    run_meta = {
        "source_path": str(source_path),
        "run_summary_path": str(run_summary_path),
        "yolo_weights": run_summary.get("yolo_weights"),
        "sam_weights": run_summary.get("sam_weights"),
        "tile_size": run_summary.get("tile_size"),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "boundary_mode": boundary_mode,
        "boundary_file": str(Path(args.boundary).resolve()) if args.boundary else None,
        "min_cluster_size": args.min_cluster_size,
        "ndvi_threshold": args.ndvi_threshold,
    }
    export_summary_json(centroids, run_meta, str(summary_path))
    _write_boundary_geojson(boundary_polygon, boundary_path)
    _write_empty_geojson(exclusion_outline_path, "exclusion_zone")
    viewer_output = generate_viewer(str(geojson_path), str(summary_path), str(viewer_path))
    _print_summary_table(centroids)
    print(f"Viewer written to: {viewer_output}")
    print(f"Open in browser: open {viewer_output}")


if __name__ == "__main__":
    main()
