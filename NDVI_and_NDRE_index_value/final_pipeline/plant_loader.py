"""
Load the ML team's plant table.

Supports the CSV produced by the ML pipeline (plants_with_ids.csv — carries
geo coords, bounding boxes and canopy mask filenames) as well as a plain
GeoJSON of points. Kept dependency-free (no geopandas).
"""

import json

import pandas as pd

from config import PLANTS_FILE, PLANTS_CRS


def load_plants(path=PLANTS_FILE):
    """
    Read the plant table into a DataFrame with one row per plant.

    Guarantees ``x`` / ``y`` columns (projected coords in the file's CRS) and
    sets ``df.attrs['crs']``.
    """
    path = str(path)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        # The ML CSV stores projected coords as geo_x / geo_y.
        if "geo_x" in df.columns and "geo_y" in df.columns:
            df["x"] = df["geo_x"]
            df["y"] = df["geo_y"]
        crs = PLANTS_CRS
    else:
        df, crs = _load_geojson(path)

    df.attrs["crs"] = crs
    print(f"[plant_loader] Loaded {len(df):,} plants  (CRS: {crs})")
    return df


def _load_geojson(path):
    with open(path, "r") as f:
        gj = json.load(f)
    crs = (gj.get("crs", {}).get("properties", {}).get("name")) or PLANTS_CRS
    rows = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        x, y = geom["coordinates"][:2]
        row = dict(feat.get("properties", {}))
        row["x"], row["y"] = x, y
        rows.append(row)
    return pd.DataFrame(rows), crs


if __name__ == "__main__":
    df = load_plants()
    print(df.head())
    print("\nColumns:", list(df.columns))
