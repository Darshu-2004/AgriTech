from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from clustering import format_sector_label


def export_geojson(centroids: list[dict], output_path: str, source_crs_wkt: str) -> None:
    """Export plant centroids and attributes as a GeoJSON FeatureCollection."""
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not centroids:
        empty_payload = {"type": "FeatureCollection", "features": []}
        destination.write_text(json.dumps(empty_payload, indent=2), encoding="utf-8")
        return

    records: list[dict] = []
    for centroid in centroids:
        properties = {key: value for key, value in centroid.items() if key not in {"geo_x", "geo_y"}}
        records.append(
            {
                **properties,
                "geometry": Point(float(centroid["geo_x"]), float(centroid["geo_y"])),
            }
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs_wkt)
    gdf.to_file(destination, driver="GeoJSON")


def export_summary_json(centroids: list[dict], run_meta: dict, output_path: str) -> None:
    """Export a JSON summary containing run metadata and the full plant ID list."""
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    sector_counter = Counter(
        format_sector_label(int(item["sector_id"])) for item in centroids if int(item["sector_id"]) != -1
    )
    summary_payload = {
        **run_meta,
        "total_plants": len(centroids),
        "plants_by_sector": dict(sorted(sector_counter.items())),
        "noise_plants": sum(1 for item in centroids if int(item["sector_id"]) == -1),
        "plant_list": centroids,
    }
    destination.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
