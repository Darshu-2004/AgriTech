from __future__ import annotations

import hashlib

from clustering import format_sector_label


def compute_geo_hash(geo_x: float, geo_y: float, precision: int = 4) -> str:
    """Compute a short stable hash from rounded geographic coordinates."""
    rounded_x = round(float(geo_x), precision)
    rounded_y = round(float(geo_y), precision)
    digest = hashlib.md5(f"{rounded_x},{rounded_y}".encode("utf-8")).hexdigest()
    return digest[:6].upper()


def assign_plant_ids(centroids: list[dict]) -> list[dict]:
    """Assign deterministic plant IDs to centroid records using sector, row, column, and hashed coordinates."""
    updated_centroids: list[dict] = []
    for centroid in centroids:
        updated = dict(centroid)
        sector_id = int(updated["sector_id"])
        row_index = int(updated["row_index"]) if updated.get("row_index") is not None else 0
        col_index = int(updated["col_index"]) if updated.get("col_index") is not None else 0
        geo_hash = compute_geo_hash(float(updated["geo_x"]), float(updated["geo_y"]))
        sector_label = format_sector_label(sector_id)
        updated["sector_label"] = sector_label
        updated["geo_hash"] = geo_hash
        updated["plant_id"] = f"PLT-{sector_label}-R{row_index:03d}-C{col_index:03d}-{geo_hash}"
        updated_centroids.append(updated)
    return updated_centroids
