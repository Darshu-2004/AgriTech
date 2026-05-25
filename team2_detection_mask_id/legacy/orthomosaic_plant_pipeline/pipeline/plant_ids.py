from __future__ import annotations

import hashlib

from .clustering import format_sector_label


def compute_geo_hash(geo_x: float, geo_y: float, precision: int = 4) -> str:
    rounded_x = round(float(geo_x), precision)
    rounded_y = round(float(geo_y), precision)
    digest = hashlib.md5(f"{rounded_x},{rounded_y}".encode("utf-8")).hexdigest()
    return digest[:6].upper()


def assign_plant_ids(plants: list[dict]) -> list[dict]:
    updated_plants: list[dict] = []
    for plant in plants:
        updated = dict(plant)
        sector_id = int(updated["sector_id"])
        row_index = int(updated["row_index"]) if updated.get("row_index") is not None else 0
        col_index = int(updated["col_index"]) if updated.get("col_index") is not None else 0
        sector_label = format_sector_label(sector_id)
        geo_hash = compute_geo_hash(float(updated["geo_x"]), float(updated["geo_y"]))

        updated["sector_label"] = sector_label
        updated["geo_hash"] = geo_hash
        updated["plant_id"] = f"PLT-{sector_label}-R{row_index:03d}-C{col_index:03d}-{geo_hash}"
        updated_plants.append(updated)
    return updated_plants
