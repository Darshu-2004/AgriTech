"""
XGBoost index gap-filling.

Features available for every (non-NOISE) plant:
  - position (geo_x, geo_y)
  - canopy morphology (area_px, canopy_area_m2, bbox_width, bbox_height)
  - sector_id
  - spatial context: mean NDVI/NDRE/OSAVI of the k nearest *measured* plants and the
    distance to them.

Three XGBRegressors (one per index) are trained on the measured plants, scored on
a held-out split, then used to predict the gaps. Predicted plants are flagged
``index_source = 'predicted'``.
"""

import numpy as np

BASE_FEATS = ["geo_x", "geo_y", "area_px", "canopy_area_m2",
              "bbox_width", "bbox_height", "sector_id"]


def _knn_context(tree, mv, coords, k, drop_self):
    """Mean neighbour value + mean distance for each coord."""
    kq = k + 1 if drop_self else k
    d, idx = tree.query(coords, k=kq)
    if drop_self:
        d, idx = d[:, 1:], idx[:, 1:]
    return mv[idx].mean(axis=1), d.mean(axis=1)


def impute_indices(df, k=8, test_frac=0.2, noise_label="NOISE"):
    """Fill missing ndvi/ndre/osavi via XGBoost. Adds ``index_source`` column."""
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
    measured = (df["ndvi"].notna() & df["ndre"].notna() & df["osavi"].notna()
                & (df["sector_label"] != noise_label)).to_numpy()
    target = (df["ndvi"].isna() & (df["sector_label"] != noise_label)
              & df.get("mask_file", df["ndvi"]).notna()).to_numpy()

    df["index_source"] = np.where(measured, "measured", None)
    if measured.sum() < 200 or target.sum() == 0:
        print(f"[xgb_impute] nothing to fill (measured={measured.sum()}, "
              f"gaps={target.sum()})")
        return df

    coords_m = df.loc[measured, ["geo_x", "geo_y"]].to_numpy()
    tree = cKDTree(coords_m)
    ndvi_m = df.loc[measured, "ndvi"].to_numpy()
    ndre_m = df.loc[measured, "ndre"].to_numpy()
    osavi_m = df.loc[measured, "osavi"].to_numpy()

    def build_X(rows, drop_self, y_col):
        coords = df.loc[rows, ["geo_x", "geo_y"]].to_numpy()
        base = df.loc[rows, feats].fillna(0.0).to_numpy()
        
        # Select spatial context based on target index to prevent circular reference during cross-val
        if y_col == "ndvi":
            kn_v, kdist = _knn_context(tree, ndvi_m, coords, k, drop_self)
            kn_r, _ = _knn_context(tree, osavi_m, coords, k, drop_self)
        elif y_col == "ndre":
            kn_v, kdist = _knn_context(tree, ndre_m, coords, k, drop_self)
            kn_r, _ = _knn_context(tree, ndvi_m, coords, k, drop_self)
        else:
            kn_v, kdist = _knn_context(tree, osavi_m, coords, k, drop_self)
            kn_r, _ = _knn_context(tree, ndvi_m, coords, k, drop_self)
            
        return np.column_stack([base, kn_v, kn_r, kdist])

    X_m_ndvi = build_X(measured, drop_self=True, y_col="ndvi")
    X_m_ndre = build_X(measured, drop_self=True, y_col="ndre")
    X_m_osavi = build_X(measured, drop_self=True, y_col="osavi")

    def fit_predict(y_col, X_m):
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
        pred = model.predict(build_X(target, drop_self=False, y_col=y_col))
        return r2, pred, model

    r2_v, pred_v, mdl_v = fit_predict("ndvi", X_m_ndvi)
    r2_r, pred_r, mdl_r = fit_predict("ndre", X_m_ndre)
    r2_o, pred_o, mdl_o = fit_predict("osavi", X_m_osavi)

    df.loc[target, "ndvi"] = np.round(pred_v, 3)
    df.loc[target, "ndre"] = np.round(pred_r, 3)
    df.loc[target, "osavi"] = np.round(pred_o, 3)
    df.loc[target, "index_source"] = "predicted"

    print(f"[xgb_impute] filled {int(target.sum()):,} plants  "
          f"(hold-out R^2: NDVI={r2_v:.3f}, NDRE={r2_r:.3f}, OSAVI={r2_o:.3f})")
    return df
