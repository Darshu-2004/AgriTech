"""
Turn raw NDVI / NDRE values into an actionable plant-health assessment.

Adapted (for an NDVI/NDRE-only dataset) from the AgriTech team_plant_analysis
health engine. Stages:

  1. KD-Tree spatial smoothing  - blend each plant's NDVI with its 8 nearest
     neighbours to suppress per-pixel noise.
  2. Vigor score (0-100)        - weighted blend of NDVI + NDRE.
  3. IsolationForest anomaly    - flag spectral outliers and blend their score.
  4. 4-tier classification      - Critical / Stressed / Moderate / Healthy,
     plus 'Out of Boundary' (no index) and 'NOISE', each with a hex color.

OSAVI and the growth-stage RandomForest from the source repo are intentionally
omitted: this dataset has no OSAVI raster and no canopy/bbox measurements.
"""

import numpy as np

# 4-tier health palette (same hex codes as the source repo for consistency).
STATUS_COLORS = {
    "Healthy":         "#2E7D32",
    "Moderate":        "#FBC02D",
    "Stressed":        "#FF5722",
    "Critical":        "#D32F2F",
    "Out of Boundary": "#1976D2",
    "NOISE":           "#757575",
}

# Vigor is scaled to the actual NDVI/NDRE distribution (robust percentiles)
# so the health spread is meaningful regardless of the absolute index scale.
ROBUST_LO_PCT, ROBUST_HI_PCT = 10, 90


def _robust_scale(x, lo_pct=ROBUST_LO_PCT, hi_pct=ROBUST_HI_PCT):
    """Map values from their p10..p90 range onto 0..100 (clipped)."""
    valid = x[~np.isnan(x)]
    if valid.size < 5:
        return np.full_like(x, 50.0)
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0) * 100


def _spatial_smooth(values, xs, ys, k=8, w_self=0.7):
    """Blend each value with the mean of its k nearest neighbours."""
    try:
        from scipy.spatial import KDTree
    except Exception:
        return values  # scipy missing -> skip smoothing gracefully

    out = values.copy()
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return out
    coords = np.column_stack([xs[mask], ys[mask]])
    vals = values[mask]
    tree = KDTree(coords)
    _, nbr = tree.query(coords, k=min(k + 1, len(coords)))
    for i in range(len(vals)):
        neigh = vals[nbr[i][1:]]
        neigh = neigh[~np.isnan(neigh)]
        if len(neigh):
            vals[i] = w_self * vals[i] + (1 - w_self) * neigh.mean()
    out[mask] = vals
    return out


def _anomaly_score(ndvi, ndre):
    """Per-plant normalized anomaly score in [0,1] via IsolationForest."""
    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        return None
    mask = ~np.isnan(ndvi)
    if mask.sum() < 30:
        return None
    X = np.column_stack([
        np.nan_to_num(ndvi[mask], nan=np.nanmedian(ndvi[mask])),
        np.nan_to_num(ndre[mask], nan=np.nanmedian(ndre[mask])),
    ])
    iso = IsolationForest(contamination=0.12, random_state=42)
    iso.fit(X)
    s = iso.decision_function(X)
    med, std = np.median(s), (np.std(s) or 0.05)
    norm = np.clip((s - (med - 2 * std)) / (3 * std), 0.0, 1.0)
    out = np.full(len(ndvi), np.nan)
    out[mask] = norm
    return out


def _classify(score):
    if np.isnan(score):
        return "Out of Boundary", STATUS_COLORS["Out of Boundary"]
    if score < 15:
        return "Critical", STATUS_COLORS["Critical"]
    if score < 40:
        return "Stressed", STATUS_COLORS["Stressed"]
    if score < 70:
        return "Moderate", STATUS_COLORS["Moderate"]
    return "Healthy", STATUS_COLORS["Healthy"]


def classify_health(df, noise_label="NOISE"):
    """
    Add health columns to ``df`` in place and return it.

    New columns: ndvi_smoothed, health_score, health_status, health_color.
    """
    ndvi = df["ndvi"].to_numpy(dtype=float)
    ndre = df["ndre"].to_numpy(dtype=float)
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)

    # 1. Spatial smoothing of NDVI.
    ndvi_sm = _spatial_smooth(ndvi, xs, ys)
    df["ndvi_smoothed"] = np.round(ndvi_sm, 3)

    # 2. Vigor score, scaled to this dataset's NDVI/NDRE distribution.
    v_ndvi = _robust_scale(ndvi_sm)
    v_ndre = _robust_scale(ndre)
    v_ndre = np.where(np.isnan(v_ndre), 50.0, v_ndre)
    vigor = 0.65 * v_ndvi + 0.35 * v_ndre

    # 3. Blend in anomaly score where available.
    anom = _anomaly_score(ndvi_sm, ndre)
    if anom is not None:
        norm = anom * 100
        score = np.where(np.isnan(norm), vigor, 0.75 * vigor + 0.25 * norm)
    else:
        score = vigor
    score = np.where(np.isnan(ndvi_sm), np.nan, score)

    # 4. Classify.
    statuses, colors = [], []
    for s in score:
        st, col = _classify(s)
        statuses.append(st)
        colors.append(col)
    df["health_score"] = np.round(score, 1)
    df["health_status"] = statuses
    df["health_color"] = colors

    # NOISE override.
    if "sector_label" in df.columns:
        nmask = df["sector_label"] == noise_label
        df.loc[nmask, "health_status"] = "NOISE"
        df.loc[nmask, "health_color"] = STATUS_COLORS["NOISE"]
        df.loc[nmask, "health_score"] = np.nan

    print("[health_classifier] status distribution:")
    for k, v in df["health_status"].value_counts().items():
        print(f"    {k:<16} {v:>7,}")
    return df
