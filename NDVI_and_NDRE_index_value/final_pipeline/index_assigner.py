"""
Assign NDVI / NDRE index values to plants.

Ties together the raster sampler and the colormap reverse-lookup: for a given
plant table and a colorized index raster, it produces the index value, the
category label, and a match-distance quality column.

When shadow masking is enabled (config.USE_SHADOW_MASKING) the true-color
orthomosaic is used to sample only sunlit leaf pixels, which avoids the
artificially-low index readings caused by shadows.
"""

import numpy as np

import config
from raster_sampler import sample_rgb_at_points, sample_index_shadow_masked


def assign_index(plants_df, tif_path, colormap, prefix):
    """
    Compute one index for every plant and return a dict of new columns.

    NOISE-labelled detections are left blank (they are weeds / false
    positives, not crops).
    """
    src_crs = plants_df.attrs.get("crs", "EPSG:4326")
    xs = plants_df["x"].values
    ys = plants_df["y"].values

    if config.USE_SHADOW_MASKING and config.ORTHO_TIF.exists():
        rgb, valid = sample_index_shadow_masked(
            tif_path, config.ORTHO_TIF, xs, ys, src_crs,
            ortho_radius=config.SHADOW_WINDOW_RADIUS_PX,
        )
        mode = "shadow-masked"
    else:
        rgb, valid = sample_rgb_at_points(tif_path, xs, ys, src_crs)
        mode = "centre-window"

    # Never assign an index to NOISE detections.
    if config.NOISE_LABEL in set(plants_df.get("sector_label", [])):
        noise = (plants_df["sector_label"] == config.NOISE_LABEL).values
        valid = valid & ~noise

    n = len(plants_df)
    values = np.full(n, np.nan)
    labels = np.array([None] * n, dtype=object)
    dists = np.full(n, np.nan)

    if valid.any():
        v, lab, d = colormap.classify(rgb[valid])
        values[valid] = v
        labels[valid] = lab
        dists[valid] = d

    covered = int(valid.sum())
    print(f"[index_assigner] {colormap.name} ({mode}): "
          f"{covered:,}/{n:,} plants matched a color "
          f"({covered / n * 100:.1f}%)")

    return {
        f"{prefix}": np.round(values, 3),
        f"{prefix}_category": labels,
        f"{prefix}_match_dist": np.round(dists, 2),
    }
