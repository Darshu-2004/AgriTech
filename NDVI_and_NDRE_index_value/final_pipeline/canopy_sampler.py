"""
Canopy-mask index sampling.

For every plant the ML team stored a binary canopy mask (outputs 2/04_masks),
sized to the plant's bounding box and positioned in the ggs-orthomosaic pixel
grid. This module averages each vegetation index over the plant's *actual*
canopy pixels — the most accurate per-plant value available.

Pipeline per plant:
  mask pixels (local) -> ggs ortho pixels -> compose affine -> index-raster
  pixels -> reverse-colormap to index values -> mean over canopy.

All index rasters are sampled in a single pass per plant so each mask PNG is
read only once.
"""

import numpy as np
import rasterio
from PIL import Image


def _composed_affine(index_tif, ggs_transform):
    """Affine mapping ggs-ortho (col,row) -> index-raster (col,row)."""
    with rasterio.open(str(index_tif)) as isrc:
        itf = isrc.transform
        rgb = isrc.read()[:3]
        H, W = rgb.shape[1], rgb.shape[2]
    A = (~itf) * ggs_transform  # applies ggs first, then index inverse
    return A, rgb, H, W


def assign_canopy_indices(plants_df, ortho_tif, masks_dir, index_specs,
                          background_rgb=(255, 255, 255), noise_label="NOISE"):
    """
    Average each index over every plant's canopy mask.

    Parameters
    ----------
    plants_df   : DataFrame with bbox_x, bbox_y, bbox_width, bbox_height,
                  mask_file, sector_label.
    ortho_tif   : the orthomosaic the masks were drawn on (for its transform).
    masks_dir   : directory containing the per-plant mask PNGs.
    index_specs : list of (tif_path, colormap, prefix).

    Returns
    -------
    dict of new columns: for each prefix -> {prefix}, {prefix}_category,
    {prefix}_match_dist, {prefix}_canopy_px.
    """
    from pathlib import Path
    masks_dir = Path(masks_dir)

    with rasterio.open(str(ortho_tif)) as osrc:
        ggs = osrc.transform

    # Pre-load every index raster + its composed affine.
    rasters = []
    for tif, cmap, prefix in index_specs:
        A, rgb, H, W = _composed_affine(tif, ggs)
        rasters.append((A, rgb, H, W, cmap, prefix))

    n = len(plants_df)
    out = {}
    for _, _, _, _, _, prefix in rasters:
        out[prefix] = np.full(n, np.nan)
        out[f"{prefix}_category"] = np.array([None] * n, dtype=object)
        out[f"{prefix}_match_dist"] = np.full(n, np.nan)
        out[f"{prefix}_canopy_px"] = np.zeros(n, dtype=int)

    bg = np.array(background_rgb, dtype=np.int16)
    req = ["bbox_x", "bbox_y", "mask_file", "sector_label"]
    missing = [c for c in req if c not in plants_df.columns]
    if missing:
        raise KeyError(f"canopy sampling needs columns {missing}")

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
        ocol = bx[i] + xs            # ggs ortho pixel coords of canopy
        orow = by[i] + ys

        any_hit = False
        for A, rgb, H, W, cmap, prefix in rasters:
            icol = A.a * ocol + A.b * orow + A.c
            irow = A.d * ocol + A.e * orow + A.f
            ic = np.round(icol).astype(int)
            ir = np.round(irow).astype(int)
            inb = (ir >= 0) & (ir < H) & (ic >= 0) & (ic < W)
            if not inb.any():
                continue
            cols_rgb = rgb[:, ir[inb], ic[inb]].T
            not_bg = ~np.all(cols_rgb.astype(np.int16) == bg, axis=1)
            kept = cols_rgb[not_bg]
            if len(kept) == 0:
                continue
            vals, _, dists = cmap.classify(kept)
            mean_val = float(vals.mean())
            out[prefix][i] = round(mean_val, 3)
            out[f"{prefix}_category"][i] = cmap.category_for_value(mean_val)
            out[f"{prefix}_match_dist"][i] = round(float(dists.mean()), 2)
            out[f"{prefix}_canopy_px"][i] = len(kept)
            any_hit = True
        if any_hit:
            matched += 1

    print(f"[canopy_sampler] {matched:,}/{n:,} plants sampled from canopy masks "
          f"({matched / n * 100:.1f}%)")
    return out
