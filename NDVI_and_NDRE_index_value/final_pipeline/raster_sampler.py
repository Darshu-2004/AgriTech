"""
Sample a colorized index GeoTIFF at the plant locations.

Handles CRS alignment between the plant points and the raster, reads a small
averaged window around each centre, and flags background / out-of-bounds
points so they can be left blank in the final CSV.
"""

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from pyproj import Transformer

from config import SAMPLE_RADIUS_PX, BACKGROUND_RGB


def _otsu_threshold(gray):
    """Pure-numpy Otsu threshold (0-255 brightness)."""
    flat = gray.ravel()
    if flat.size == 0:
        return 50.0
    hist, edges = np.histogram(flat, bins=256, range=(0, 256))
    centers = (edges[:-1] + edges[1:]) / 2.0
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    w1 = np.where(w1 == 0, 1e-5, w1)
    w2 = np.where(w2 == 0, 1e-5, w2)
    m1 = np.cumsum(hist * centers) / w1
    m2 = (np.cumsum((hist * centers)[::-1]) / w2[::-1])[::-1]
    var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    if var.size == 0:
        return 50.0
    return float(centers[np.argmax(var)])


def sample_rgb_at_points(tif_path, xs, ys, src_crs,
                         radius=SAMPLE_RADIUS_PX,
                         background_rgb=BACKGROUND_RGB):
    """
    Return the representative RGB at each (x, y) point in a colorized raster.

    Parameters
    ----------
    tif_path       : path to a 3/4-band uint8 GeoTIFF.
    xs, ys         : sequences of point coordinates in ``src_crs``.
    src_crs        : CRS string of the incoming points (e.g. "EPSG:32647").
    radius         : half-size of the averaging window in pixels.
    background_rgb : RGB treated as no-data (excluded from the window mean).

    Returns
    -------
    rgb   : (N, 3) float array; NaN rows = point had no valid pixel.
    valid : (N,) bool mask, True where a usable color was found.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)

    with rasterio.open(str(tif_path)) as src:
        # Reproject points into the raster CRS if they differ.
        if src.crs and src_crs and str(src.crs) != str(src_crs):
            tf = Transformer.from_crs(src_crs, src.crs, always_xy=True)
            xs, ys = tf.transform(xs, ys)

        bands = src.read()                      # (bands, H, W) uint8
        rgb_img = bands[:3]                      # (3, H, W)
        H, W = rgb_img.shape[1], rgb_img.shape[2]

        rows, cols = rowcol(src.transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)

    bg = np.array(background_rgb, dtype=np.int16)
    out_rgb = np.full((n, 3), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        if r < 0 or r >= H or c < 0 or c >= W:
            continue  # outside raster coverage
        r0, r1 = max(0, r - radius), min(H, r + radius + 1)
        c0, c1 = max(0, c - radius), min(W, c + radius + 1)
        window = rgb_img[:, r0:r1, c0:c1].reshape(3, -1).T  # (px, 3)

        # Drop background pixels from the window.
        not_bg = ~np.all(window.astype(np.int16) == bg, axis=1)
        kept = window[not_bg]
        if len(kept) == 0:
            continue
        out_rgb[i] = kept.mean(axis=0)
        valid[i] = True

    return out_rgb, valid


def sample_index_shadow_masked(index_tif, ortho_tif, xs, ys, src_crs,
                               ortho_radius=5, brightness_floor=50.0,
                               background_rgb=BACKGROUND_RGB,
                               ortho_max_dim=6000):
    """
    Sample a colorized index raster using Otsu shadow masking.

    For each plant a small window of the true-color orthomosaic is taken,
    Otsu-thresholded into sunlit vs shadowed pixels, and only the *sunlit*
    leaf pixels are projected into the index raster and averaged. This avoids
    the artificially-low index readings that shadows produce.

    Returns
    -------
    rgb   : (N, 3) float, mean sunlit color (NaN where nothing usable).
    valid : (N,) bool.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)

    # --- orthomosaic: brightness + transform (read downsampled to cap RAM) ---
    from rasterio.enums import Resampling
    from affine import Affine
    with rasterio.open(str(ortho_tif)) as osrc:
        if osrc.crs and src_crs and str(osrc.crs) != str(src_crs):
            tf = Transformer.from_crs(src_crs, osrc.crs, always_xy=True)
            oxs, oys = tf.transform(xs, ys)
        else:
            oxs, oys = xs, ys
        scale = min(1.0, ortho_max_dim / max(osrc.width, osrc.height))
        oW, oH = int(osrc.width * scale), int(osrc.height * scale)
        obands = osrc.read(out_shape=(osrc.count, oH, oW),
                           resampling=Resampling.bilinear)
        o_tf = osrc.transform * Affine.scale(osrc.width / oW, osrc.height / oH)
        o_crs = osrc.crs
        o_rows, o_cols = rowcol(o_tf, oxs, oys)
    o_rows = np.asarray(o_rows); o_cols = np.asarray(o_cols)
    bright = (0.299 * obands[0] + 0.587 * obands[1] +
              0.114 * obands[2]).astype(np.float32)

    # --- index raster ---
    with rasterio.open(str(index_tif)) as isrc:
        if isrc.crs and str(isrc.crs) != str(o_crs):
            tf2 = Transformer.from_crs(o_crs, isrc.crs, always_xy=True)
        else:
            tf2 = None
        ibands = isrc.read()[:3]
        i_tf = isrc.transform
        iH, iW = ibands.shape[1], ibands.shape[2]

    bg = np.array(background_rgb, dtype=np.int16)
    out_rgb = np.full((n, 3), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for k in range(n):
        r, c = int(o_rows[k]), int(o_cols[k])
        if not (0 <= r < oH and 0 <= c < oW):
            continue
        r0, r1 = max(0, r - ortho_radius), min(oH, r + ortho_radius + 1)
        c0, c1 = max(0, c - ortho_radius), min(oW, c + ortho_radius + 1)
        win = bright[r0:r1, c0:c1]
        if win.size == 0:
            continue
        thresh = max(brightness_floor, _otsu_threshold(win))
        ys_idx, xs_idx = np.where(win > thresh)
        if len(ys_idx) == 0:               # all shadow -> fall back to centre
            ys_idx, xs_idx = np.array([r - r0]), np.array([c - c0])
        abs_r = ys_idx + r0
        abs_c = xs_idx + c0

        # ortho pixel centres -> geo
        gx, gy = xy(o_tf, abs_r, abs_c)
        gx = np.asarray(gx); gy = np.asarray(gy)
        if tf2 is not None:
            gx, gy = tf2.transform(gx, gy)

        # geo -> index raster pixels
        irow, icol = rowcol(i_tf, gx, gy)
        irow = np.asarray(irow); icol = np.asarray(icol)
        inb = (irow >= 0) & (irow < iH) & (icol >= 0) & (icol < iW)
        if not inb.any():
            continue
        cols_rgb = ibands[:, irow[inb], icol[inb]].T  # (m, 3)
        not_bg = ~np.all(cols_rgb.astype(np.int16) == bg, axis=1)
        kept = cols_rgb[not_bg]
        if len(kept) == 0:
            continue
        out_rgb[k] = kept.mean(axis=0)
        valid[k] = True

    return out_rgb, valid
