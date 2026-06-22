"""
Extract a GeoTIFF's metadata to a JSON sidecar next to the file.

Usage:
    python extract_tif_metadata.py <path/to/file.tif> [more.tif ...]

Default (no args): the OSAVI map in ../maps/.
"""

import sys
import json
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


def extract(tif_path):
    tif_path = Path(tif_path)
    with rasterio.open(str(tif_path)) as s:
        t = s.transform
        west, south, east, north = (s.bounds.left, s.bounds.bottom,
                                    s.bounds.right, s.bounds.top)
        try:
            ll = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
        except Exception:
            ll = None
        meta = {
            "file": tif_path.name,
            "file_size_bytes": tif_path.stat().st_size,
            "driver": s.driver,
            "width": s.width,
            "height": s.height,
            "band_count": s.count,
            "dtype": s.dtypes[0],
            "nodata": s.nodata,
            "crs": str(s.crs),
            "crs_epsg": s.crs.to_epsg() if s.crs else None,
            "crs_wkt": s.crs.to_wkt() if s.crs else None,
            "transform": list(t)[:6],
            "pixel_size_x_m": abs(t.a),
            "pixel_size_y_m": abs(t.e),
            "gsd_cm": round(abs(t.a) * 100, 4),
            "bounds_utm": {"left": west, "bottom": south,
                           "right": east, "top": north},
            "bounds_lonlat": (
                {"west": ll[0], "south": ll[1], "east": ll[2], "north": ll[3]}
                if ll else None),
            "extent_m": {"width": round(east - west, 3),
                         "height": round(north - south, 3)},
            "compression": (s.profile.get("compress")),
            "tags": s.tags(),
            "band_tags": {b: s.tags(b) for b in range(1, s.count + 1)},
            "band_descriptions": list(s.descriptions),
            "tag_namespaces": {ns: s.tags(ns=ns) for ns in s.tag_namespaces()},
        }
    out = tif_path.with_name(tif_path.stem + "_metadata.json")
    out.write_text(json.dumps(meta, indent=2))
    print(f"[metadata] {tif_path.name}: {s.width}x{s.height}, {s.count} bands, "
          f"{meta['gsd_cm']} cm GSD, CRS {meta['crs']}")
    print(f"           saved -> {out}")
    return out


def main():
    args = sys.argv[1:]
    if not args:
        root = Path(__file__).resolve().parent.parent
        args = [root / "maps" / "Task-of-2026-03-22T063838646Z-orthophoto-OSAVI.tif"]
    for a in args:
        extract(a)


if __name__ == "__main__":
    main()
