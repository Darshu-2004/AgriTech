"""
load_to_postgis.py
------------------
Loads plant health GeoJSON produced by extract_ndvi.py or
extract_osavi.py into PostgreSQL + PostGIS.

Usage:
    python scripts/load_to_postgis.py \
        --input output/plants_ndvi.json \
        --dsn "postgresql://user:pass@localhost:5432/mydb"

or

    python scripts/load_to_postgis.py \
        --input output/plants_osavi.json \
        --dsn "postgresql://user:pass@localhost:5432/mydb"

Requirements:
    pip install psycopg2-binary pyproj
"""

import argparse
import json
import sys

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit(
        "psycopg2-binary is required: pip install psycopg2-binary"
    )

try:
    from pyproj import Transformer
    HAVE_PROJ = True
except ImportError:
    HAVE_PROJ = False


# ─────────────────────────────────────────────────────────────
# Reprojection
# ─────────────────────────────────────────────────────────────

def reproject_point(x: float, y: float, from_crs: str):
    """
    Reproject coordinates to WGS84 (EPSG:4326)
    """

    if (
        not HAVE_PROJ
        or from_crs is None
        or from_crs.lower() == "epsg:4326"
    ):
        return x, y

    transformer = Transformer.from_crs(
        from_crs,
        "EPSG:4326",
        always_xy=True
    )

    return transformer.transform(x, y)


# ─────────────────────────────────────────────────────────────
# SQL
# ─────────────────────────────────────────────────────────────

INSERT_SQL = """
INSERT INTO plants (
    plant_id,
    geo_hash,
    sector_id,
    sector_label,
    row_index,
    col_index,
    pixel_x,
    pixel_y,
    area_px,
    confidence,

    ndvi,
    osavi,

    health_status,
    health_color,
    raster_crs,

    geom

) VALUES (

    %(plant_id)s,
    %(geo_hash)s,
    %(sector_id)s,
    %(sector_label)s,
    %(row_index)s,
    %(col_index)s,
    %(pixel_x)s,
    %(pixel_y)s,
    %(area_px)s,
    %(confidence)s,

    %(ndvi)s,
    %(osavi)s,

    %(health_status)s,
    %(health_color)s,
    %(raster_crs)s,

    ST_SetSRID(
        ST_MakePoint(%(lon)s, %(lat)s),
        4326
    )
)

ON CONFLICT DO NOTHING;
"""

BATCH_SIZE = 1000


# ─────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────

def load(input_path, dsn, table, truncate):

    print(f"[1/3] Reading {input_path} ...")

    with open(input_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])

    print(f"      {len(features):,} features found")

    print("[2/3] Connecting to PostgreSQL ...")

    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    if truncate:
        print(f"      Truncating '{table}'")
        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY;")
        conn.commit()

    rows = []
    skipped = 0

    for feat in features:

        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        coords = geom.get("coordinates", [])

        if len(coords) < 2:
            skipped += 1
            continue

        from_crs = props.get(
            "raster_crs",
            "EPSG:4326"
        )

        lon, lat = reproject_point(
            float(coords[0]),
            float(coords[1]),
            from_crs
        )

        rows.append({

            "plant_id": props.get("plant_id"),
            "geo_hash": props.get("geo_hash"),

            "sector_id": props.get("sector_id"),
            "sector_label": props.get("sector_label"),

            "row_index": props.get("row_index"),
            "col_index": props.get("col_index"),

            "pixel_x": props.get("pixel_x"),
            "pixel_y": props.get("pixel_y"),

            "area_px": props.get("area_px"),
            "confidence": props.get("confidence"),

            # NDVI pipeline
            "ndvi": props.get("ndvi"),

            # OSAVI pipeline
            "osavi": props.get("osavi"),

            "health_status": props.get("health_status"),
            "health_color": props.get("health_color"),

            "raster_crs": from_crs,

            "lon": lon,
            "lat": lat
        })

    print(
        f"[3/3] Inserting {len(rows):,} rows "
        f"in batches of {BATCH_SIZE}"
    )

    for i in range(0, len(rows), BATCH_SIZE):

        batch = rows[i:i + BATCH_SIZE]

        psycopg2.extras.execute_batch(
            cur,
            INSERT_SQL,
            batch,
            page_size=BATCH_SIZE
        )

        conn.commit()

        inserted = min(
            i + BATCH_SIZE,
            len(rows)
        )

        print(
            f"      {inserted:,} / {len(rows):,} inserted",
            end="\r"
        )

    print(
        f"\n✅ Loaded {len(rows):,} plants "
        f"(skipped {skipped})"
    )

    cur.close()
    conn.close()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        description="Load plant health GeoJSON into PostGIS."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input GeoJSON"
    )

    parser.add_argument(
        "--dsn",
        required=True,
        help="PostgreSQL connection string"
    )

    parser.add_argument(
        "--table",
        default="plants",
        help="Target table"
    )

    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Clear table before loading"
    )

    args = parser.parse_args()

    load(
        args.input,
        args.dsn,
        args.table,
        args.truncate
    )


if __name__ == "__main__":
    main()