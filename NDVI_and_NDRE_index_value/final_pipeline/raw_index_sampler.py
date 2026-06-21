"""
Canopy-mean of the REAL (continuous) NDVI / NDRE values.

The colorized index GeoTIFFs only encode binned values. The true per-pixel
index values live in the multispectral export
``ndvi_ndre_Unknown_2026-06-10.csv`` (lat, lon, ndvi, ndre, plus chlorophyll /
stress / nsi). That grid is sparse (~25 cm), so instead of point-in-canopy
intersection we:

  1. Build a KD-Tree over the raw index points (in UTM).
  2. For each plant, take its canopy-mask pixels, look up the nearest real
     index value for each, and average -> an area-weighted canopy mean of the
     actual index values.

Plants whose canopy has no raw sample within ``max_match_m`` are left blank
(outside the multispectral coverage).
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from PIL import Image


# Extra per-pixel metrics to average alongside ndvi / ndre, if present.
EXTRA_COLS = ["nsi", "chlorophyll_idx", "stress_idx"]


def _load_grid(csv_path, plants_crs):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ndvi", "ndre", "lat", "lon"])
    tf = Transformer.from_crs("EPSG:4326", plants_crs, always_xy=True)
    gx, gy = tf.transform(df["lon"].values, df["lat"].values)
    cols = ["ndvi", "ndre"] + [c for c in EXTRA_COLS if c in df.columns]
    vals = df[cols].to_numpy(dtype=float)
    return np.column_stack([gx, gy]), vals, cols


def assign_raw_canopy_indices(plants_df, ortho_tif, masks_dir, csv_path,
                              plants_crs="EPSG:32647", max_match_m=0.5,
                              max_px_per_plant=200, noise_label="NOISE",
                              ndvi_cmap=None, ndre_cmap=None):
    """
    Return per-plant canopy means of the real index values as new columns.
    """
    from pathlib import Path
    from scipy.spatial import cKDTree
    masks_dir = Path(masks_dir)

    pts, vals, cols = _load_grid(csv_path, plants_crs)
    tree = cKDTree(pts)
    print(f"[raw_index] loaded {len(pts):,} real index points; "
          f"matching canopies (<= {max_match_m} m)...")

    with rasterio.open(str(ortho_tif)) as osrc:
        ggs = osrc.transform

    n = len(plants_df)
    out = {c: np.full(n, np.nan) for c in cols}
    out["ndvi_canopy_px"] = np.zeros(n, dtype=int)
    out["ndvi_category"] = np.array([None] * n, dtype=object)
    out["ndre_category"] = np.array([None] * n, dtype=object)

    bx = plants_df["bbox_x"].to_numpy()
    by = plants_df["bbox_y"].to_numpy()
    mf = plants_df["mask_file"].to_numpy()
    lab = plants_df["sector_label"].to_numpy()

    matched = 0
    for i in range(n):
        if lab[i] == noise_label:
            continue
        mpath = masks_dir / str(mf[i])
        if not mpath.exists():
            continue
        mask = np.asarray(Image.open(mpath))
        if mask.ndim == 3:
            mask = mask[..., 0]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        if len(xs) > max_px_per_plant:          # subsample for speed
            sel = np.linspace(0, len(xs) - 1, max_px_per_plant).astype(int)
            xs, ys = xs[sel], ys[sel]

        gx, gy = xy(ggs, by[i] + ys, bx[i] + xs)   # canopy pixel -> geo
        q = np.column_stack([np.asarray(gx), np.asarray(gy)])
        dist, idx = tree.query(q, k=1)
        ok = dist <= max_match_m
        if not ok.any():
            continue
        sub = vals[idx[ok]]
        means = np.nanmean(sub, axis=0)
        for j, c in enumerate(cols):
            out[c][i] = round(float(means[j]), 4)
        out["ndvi_canopy_px"][i] = int(ok.sum())
        if ndvi_cmap is not None:
            out["ndvi_category"][i] = ndvi_cmap.category_for_value(out["ndvi"][i])
        if ndre_cmap is not None:
            out["ndre_category"][i] = ndre_cmap.category_for_value(out["ndre"][i])
        matched += 1

    print(f"[raw_index] {matched:,}/{n:,} plants got a real canopy-mean "
          f"({matched / n * 100:.1f}%)")
    return out
