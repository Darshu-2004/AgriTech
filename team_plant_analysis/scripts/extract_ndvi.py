"""
extract_ndvi.py
---------------
Extracts NDVI values for each detected plant from a visual NDVI raster,
classifies health status, assigns colour codes, and writes a PostGIS-ready
JSON file.

Usage:
    python scripts/extract_ndvi.py \
        --plants   data/plants.geojson \
        --ndvi     data/plant_health.tif \
        --output   output/plants_ndvi.json \
        [--preview output/preview.png] \
        [--ortho   data/orthomosaic.tif]

NDVI proxy:
    The visual NDVI raster encodes health as colour:
        • Healthy  → green  (G >> R)  → NDVI ≈ +0.4 … +1.0
        • Moderate → yellow (G ≈ R)   → NDVI ≈  0.0 … +0.4
        • Diseased → red    (R >> G)   → NDVI ≈ -1.0 …  0.0

    Formula: NDVI_proxy = (G − R) / (G + R)

Thresholds (adjustable via --healthy / --moderate flags):
    Healthy   ≥ 0.40   → #2ECC71  (green)
    Moderate  ≥ 0.10   → #F39C12  (amber)
    Diseased   < 0.10  → #E74C3C  (red)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# ── optional progress bar ─────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

# ── health classification ─────────────────────────────────────────────────────
DEFAULT_HEALTHY_THRESH  = 0.40
DEFAULT_MODERATE_THRESH = 0.10

HEALTH_COLORS = {
    "healthy":  "#2ECC71",   # green
    "moderate": "#F39C12",   # amber
    "diseased": "#E74C3C",   # red
}


def classify(ndvi: float, healthy_t: float, moderate_t: float) -> tuple[str, str]:
    """Return (status, hex_colour) for a given NDVI value."""
    if ndvi >= healthy_t:
        return "healthy",  HEALTH_COLORS["healthy"]
    elif ndvi >= moderate_t:
        return "moderate", HEALTH_COLORS["moderate"]
    else:
        return "diseased", HEALTH_COLORS["diseased"]


# ── NDVI extraction ───────────────────────────────────────────────────────────
def extract_ndvi_at_point(
    src: rasterio.DatasetReader,
    red_band: np.ndarray,
    green_band: np.ndarray,
    lon: float,
    lat: float,
) -> float | None:
    """
    Sample the NDVI proxy at a geographic point.
    Returns None when the point falls outside the raster or on a no-data pixel.
    """
    try:
        row, col = src.index(lon, lat)
    except Exception:
        return None

    if not (0 <= row < src.height and 0 <= col < src.width):
        return None

    r = float(red_band[row, col])
    g = float(green_band[row, col])
    denom = g + r
    if denom == 0:
        return None

    return (g - r) / denom


# ── main processing ───────────────────────────────────────────────────────────
def process(
    plants_path: str,
    ndvi_path: str,
    output_path: str,
    healthy_t: float,
    moderate_t: float,
    preview_path: str | None,
    ortho_path: str | None,
) -> None:

    # ── load plants ───────────────────────────────────────────────────────────
    print(f"[1/4] Loading plants from {plants_path} …")
    with open(plants_path) as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    print(f"      {len(features):,} plant detections found.")

    # ── load raster bands (full read is fast enough for this resolution) ──────
    print(f"[2/4] Reading NDVI raster {ndvi_path} …")
    with rasterio.open(ndvi_path) as src:
        raster_crs = str(src.crs)
        red_band   = src.read(1).astype(np.float32)
        green_band = src.read(2).astype(np.float32)
        # suppress 0/0 warnings
        np.seterr(divide="ignore", invalid="ignore")

        # ── extract NDVI per plant ────────────────────────────────────────────
        print(f"[3/4] Extracting NDVI for {len(features):,} plants …")
        output_features = []
        stats = {"healthy": 0, "moderate": 0, "diseased": 0, "skipped": 0}

        iterator = (
            tqdm(features, unit="plant", desc="      Extracting") if _TQDM
            else features
        )

        for feat in iterator:
            geom = feat.get("geometry", {})
            props = dict(feat.get("properties", {}))

            if geom.get("type") != "Point":
                stats["skipped"] += 1
                continue

            lon, lat = geom["coordinates"]

            ndvi = extract_ndvi_at_point(src, red_band, green_band, lon, lat)
            if ndvi is None:
                stats["skipped"] += 1
                continue

            ndvi = round(float(ndvi), 6)
            status, color = classify(ndvi, healthy_t, moderate_t)
            stats[status] += 1

            props.update(
                ndvi=ndvi,
                health_status=status,
                health_color=color,
                raster_crs=raster_crs,
            )

            output_features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": props,
                }
            )

        # ── generate preview ──────────────────────────────────────────────────
        if preview_path:
            _generate_preview(
                ndvi_path=ndvi_path,
                ortho_path=ortho_path,
                features=output_features,
                preview_path=preview_path,
                src_ref=src,
            )

    # ── write output ──────────────────────────────────────────────────────────
    print(f"[4/4] Writing output → {output_path} …")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_geojson = {"type": "FeatureCollection", "features": output_features}
    with open(output_path, "w") as f:
        json.dump(output_geojson, f, indent=2)

    total = sum(v for k, v in stats.items() if k != "skipped")
    print("\n✅  Done!")
    print(f"   Healthy   : {stats['healthy']:>6,}  ({stats['healthy']/max(total,1)*100:.1f} %)")
    print(f"   Moderate  : {stats['moderate']:>6,}  ({stats['moderate']/max(total,1)*100:.1f} %)")
    print(f"   Diseased  : {stats['diseased']:>6,}  ({stats['diseased']/max(total,1)*100:.1f} %)")
    print(f"   Skipped   : {stats['skipped']:>6,}")
    print(f"   Output    : {output_path}")


# ── preview helper ────────────────────────────────────────────────────────────
def _generate_preview(
    ndvi_path: str,
    ortho_path: str | None,
    features: list,
    preview_path: str,
    src_ref: rasterio.DatasetReader,
) -> None:
    """Render a downsampled NDVI image with plant dots overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from rasterio.plot import show as rshow

    print(f"      Generating preview → {preview_path} …")

    # choose background: ortho if provided, else NDVI
    bg_path = ortho_path if ortho_path else ndvi_path

    # downsample factor so image fits in memory for plotting
    DOWNSAMPLE = 4

    with rasterio.open(bg_path) as bg:
        # read downsampled RGB
        h = bg.height // DOWNSAMPLE
        w = bg.width  // DOWNSAMPLE
        r = bg.read(1, out=np.empty((h, w), dtype=np.uint8))
        g = bg.read(2, out=np.empty((h, w), dtype=np.uint8))
        b_ch = bg.read(3, out=np.empty((h, w), dtype=np.uint8))
        rgb = np.stack([r, g, b_ch], axis=-1)

        # raster extent for imshow
        left, bottom, right, top = bg.bounds

    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.imshow(
        rgb,
        extent=[left, right, bottom, top],
        origin="upper",
        interpolation="bilinear",
    )

    # scatter plant dots coloured by health
    color_map = {
        "healthy":  "#2ECC71",
        "moderate": "#F39C12",
        "diseased": "#E74C3C",
    }
    DOT_SIZE = 12

    xs = {"healthy": [], "moderate": [], "diseased": []}
    ys = {"healthy": [], "moderate": [], "diseased": []}

    for feat in features:
        status = feat["properties"].get("health_status", "diseased")
        lon, lat = feat["geometry"]["coordinates"]
        xs[status].append(lon)
        ys[status].append(lat)

    for status, col in color_map.items():
        if xs[status]:
            ax.scatter(
                xs[status], ys[status],
                c=col, s=DOT_SIZE, linewidths=0,
                alpha=0.8, label=status.capitalize(), zorder=5,
            )

    # legend
    legend_patches = [
        mpatches.Patch(color=c, label=f"{s.capitalize()} ({len(xs[s]):,})")
        for s, c in color_map.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.85)

    ax.set_title("NDVI Plant Health Preview", fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Preview saved → {preview_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Extract NDVI values for detected plants.")
    p.add_argument("--plants",   required=True, help="Path to plants GeoJSON")
    p.add_argument("--ndvi",     required=True, help="Path to NDVI / plant_health TIFF")
    p.add_argument("--output",   required=True, help="Output GeoJSON path")
    p.add_argument("--ortho",    default=None,  help="Optional orthomosaic TIFF for preview background")
    p.add_argument("--preview",  default=None,  help="Output preview PNG path")
    p.add_argument("--healthy",  type=float, default=DEFAULT_HEALTHY_THRESH,
                   help=f"NDVI threshold for healthy (default {DEFAULT_HEALTHY_THRESH})")
    p.add_argument("--moderate", type=float, default=DEFAULT_MODERATE_THRESH,
                   help=f"NDVI threshold for moderate (default {DEFAULT_MODERATE_THRESH})")
    args = p.parse_args()

    process(
        plants_path  = args.plants,
        ndvi_path    = args.ndvi,
        output_path  = args.output,
        healthy_t    = args.healthy,
        moderate_t   = args.moderate,
        preview_path = args.preview,
        ortho_path   = args.ortho,
    )


if __name__ == "__main__":
    main()
