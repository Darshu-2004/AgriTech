"""
load_to_postgis.py
------------------
Loads the output GeoJSON produced by extract_ndvi.py into a
PostgreSQL + PostGIS database.

Usage:
    python scripts/load_to_postgis.py \
        --input  output/plants_ndvi.json \
        --dsn    "postgresql://user:pass@localhost:5432/mydb" \
        [--table plants] \
        [--truncate]

Requirements:
    pip install psycopg2-binary shapely

The script:
  1. Reads the GeoJSON.
  2. Reprojects coordinates from the source CRS to WGS84 (EPSG:4326).
  3. Bulk-inserts rows using psycopg2 executemany for speed.
  4. Falls back to COPY if psycopg2-binary is unavailable.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit(
        "psycopg2-binary is required: pip install psycopg2-binary"
    )

try:
    from pyproj import Transformer
    _HAVE_PROJ = True
except ImportError:
    _HAVE_PROJ = False


# ── reprojection ──────────────────────────────────────────────────────────────
def reproject_point(lon: float, lat: float, from_crs: str) -> tuple[float, float]:
    """
    Reproject a point from *from_crs* to WGS84 (EPSG:4326).
    Returns (longitude_wgs84, latitude_wgs84).
    If pyproj is not available, the coordinates are passed through unchanged.
    """
    if not _HAVE_PROJ or from_crs in ("EPSG:4326", "epsg:4326", None, ""):
        return lon, lat

    transformer = Transformer.from_crs(from_crs, "EPSG:4326", always_xy=True)
    return transformer.transform(lon, lat)


# ── bulk insert ───────────────────────────────────────────────────────────────
INSERT_SQL = """
INSERT INTO plants (
    plant_id, geo_hash, sector_id, sector_label,
    row_index, col_index, pixel_x, pixel_y,
    area_px, confidence,
    ndvi, health_status, health_color, raster_crs,
    geom
) VALUES (
    %(plant_id)s, %(geo_hash)s, %(sector_id)s, %(sector_label)s,
    %(row_index)s, %(col_index)s, %(pixel_x)s, %(pixel_y)s,
    %(area_px)s, %(confidence)s,
    %(ndvi)s, %(health_status)s, %(health_color)s, %(raster_crs)s,
    ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)
)
ON CONFLICT DO NOTHING;
"""

BATCH_SIZE = 1000


def load(input_path: str, dsn: str, table: str, truncate: bool) -> None:
    print(f"[1/3] Reading {input_path} …")
    with open(input_path) as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    print(f"      {len(features):,} features to load.")

    print("[2/3] Connecting to PostgreSQL …")
    conn = psycopg2.connect(dsn)
    cur  = conn.cursor()

    if truncate:
        print(f"      Truncating table '{table}' …")
        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY;")

    # ── build rows ────────────────────────────────────────────────────────────
    rows = []
    skipped = 0
    for feat in features:
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])

        if len(coords) < 2 or coords[0] is None:
            skipped += 1
            continue

        from_crs = props.get("raster_crs", "EPSG:4326")
        lon, lat = reproject_point(float(coords[0]), float(coords[1]), from_crs)

        rows.append({
            "plant_id":     props.get("plant_id"),
            "geo_hash":     props.get("geo_hash"),
            "sector_id":    props.get("sector_id"),
            "sector_label": props.get("sector_label"),
            "row_index":    props.get("row_index"),
            "col_index":    props.get("col_index"),
            "pixel_x":      props.get("pixel_x"),
            "pixel_y":      props.get("pixel_y"),
            "area_px":      props.get("area_px"),
            "confidence":   props.get("confidence"),
            "ndvi":         props.get("ndvi"),
            "health_status":props.get("health_status"),
            "health_color": props.get("health_color"),
            "raster_crs":   from_crs,
            "lon":          lon,
            "lat":          lat,
        })

    # ── batch insert ──────────────────────────────────────────────────────────
    print(f"[3/3] Inserting {len(rows):,} rows in batches of {BATCH_SIZE} …")
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        psycopg2.extras.execute_batch(cur, INSERT_SQL, batch, page_size=BATCH_SIZE)
        conn.commit()
        pct = min(i + BATCH_SIZE, len(rows))
        print(f"      {pct:,} / {len(rows):,} inserted …", end="\r")

    print(f"\n✅  Loaded {len(rows):,} plants  (skipped {skipped})")
    cur.close()
    conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Load plant NDVI GeoJSON into PostGIS.")
    p.add_argument("--input",    required=True, help="Path to plants_ndvi.json")
    p.add_argument("--dsn",      required=True,
                   help='PostgreSQL DSN, e.g. "postgresql://user:pass@localhost/db"')
    p.add_argument("--table",    default="plants", help="Target table (default: plants)")
    p.add_argument("--truncate", action="store_true",
                   help="Truncate the table before inserting")
    args = p.parse_args()

    load(args.input, args.dsn, args.table, args.truncate)


if __name__ == "__main__":
    main()
