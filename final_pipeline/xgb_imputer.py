"""
XGBoost index gap-filling.

The colorized maps cover ~96% of the plants; the rest (and any future sparse
survey) have no NDVI/NDRE. Those gaps can be filled by supervised learning:
NDVI/NDRE are spatially + morphologically structured, so a gradient-boosted
model trained on the *measured* plants can predict the missing ones.

Features available for every (non-NOISE) plant:
  - position (x, y)
  - canopy morphology (area_px, canopy_area_m2, bbox_width, bbox_height)
  - sector_id
  - spatial context: mean NDVI/NDRE of the k nearest *measured* plants and the
    distance to them (strong predictors — neighbouring plants are similar).

Two XGBRegressors (one per index) are trained on the measured plants, scored on
a held-out split, then used to predict the gaps. Predicted plants are flagged
``index_source = 'predicted'`` so they are never confused with measurements.
"""

import numpy as np

BASE_FEATS = ["x", "y", "area_px", "canopy_area_m2",
              "bbox_width", "bbox_height", "sector_id"]


def _knn_context(tree, mv, coords, k, drop_self):
    """Mean neighbour value + mean distance for each coord."""
    kq = k + 1 if drop_self else k
    d, idx = tree.query(coords, k=kq)
    if drop_self:
        d, idx = d[:, 1:], idx[:, 1:]
    return mv[idx].mean(axis=1), d.mean(axis=1)


def impute_indices(df, k=8, test_frac=0.2, noise_label="NOISE"):
    """Fill missing ndvi/ndre via XGBoost. Adds ``index_source`` column."""
    try:
        from xgboost import XGBRegressor
        from scipy.spatial import cKDTree
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
    except Exception as e:
        print(f"[xgb_impute] skipped ({e})")
        df["index_source"] = np.where(df["ndvi"].notna(), "measured", None)
        return df

    feats = [f for f in BASE_FEATS if f in df.columns]
    measured = (df["ndvi"].notna() & df["ndre"].notna()
                & (df["sector_label"] != noise_label)).to_numpy()
    target = (df["ndvi"].isna() & (df["sector_label"] != noise_label)
              & df.get("mask_file", df["ndvi"]).notna()).to_numpy()

    df["index_source"] = np.where(measured, "measured", None)
    if measured.sum() < 200 or target.sum() == 0:
        print(f"[xgb_impute] nothing to fill (measured={measured.sum()}, "
              f"gaps={target.sum()})")
        return df

    coords_m = df.loc[measured, ["x", "y"]].to_numpy()
    tree = cKDTree(coords_m)
    ndvi_m = df.loc[measured, "ndvi"].to_numpy()
    ndre_m = df.loc[measured, "ndre"].to_numpy()

    def build_X(rows, drop_self):
        coords = df.loc[rows, ["x", "y"]].to_numpy()
        base = df.loc[rows, feats].fillna(0.0).to_numpy()
        kn_v, kdist = _knn_context(tree, ndvi_m, coords, k, drop_self)
        kn_r, _ = _knn_context(tree, ndre_m, coords, k, drop_self)
        return np.column_stack([base, kn_v, kn_r, kdist])

    feat_names = feats + ["knn_ndvi", "knn_ndre", "knn_dist"]
    X_m = build_X(measured, drop_self=True)

    def fit_predict(y_col):
        y = df.loc[measured, y_col].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(
            X_m, y, test_size=test_frac, random_state=42)
        model = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=4)
        model.fit(Xtr, ytr)
        r2 = r2_score(yte, model.predict(Xte))
        model.fit(X_m, y)               # refit on all measured
        pred = model.predict(build_X(target, drop_self=False))
        return r2, pred, model

    r2_v, pred_v, mdl_v = fit_predict("ndvi")
    r2_r, pred_r, mdl_r = fit_predict("ndre")

    df.loc[target, "ndvi"] = np.round(pred_v, 3)
    df.loc[target, "ndre"] = np.round(pred_r, 3)
    df.loc[target, "index_source"] = "predicted"

    print(f"[xgb_impute] filled {int(target.sum()):,} plants  "
          f"(hold-out R^2: NDVI={r2_v:.3f}, NDRE={r2_r:.3f})")
    imp = sorted(zip(feat_names, mdl_v.feature_importances_),
                 key=lambda t: -t[1])[:5]
    print("[xgb_impute] top NDVI(biomass) drivers: "
          + ", ".join(f"{n}={v:.2f}" for n, v in imp))
    return df
