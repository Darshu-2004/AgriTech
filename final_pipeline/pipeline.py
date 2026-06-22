"""
Orchestrates the full plant-health index pipeline:

    plants.geojson  +  NDVI raster  +  NDRE raster
            │
            ▼
    plant_health_indices.csv   (one row per plant, with NDVI & NDRE)

Running this module (or run.py) executes every stage end-to-end.
"""

import numpy as np
import rasterio
from pyproj import Transformer

import config
from plant_loader import load_plants
from index_assigner import assign_index
from canopy_sampler import assign_canopy_indices
from health_classifier import classify_health
from colormaps import NDVI_CMAP, NDRE_CMAP, OSAVI_CMAP


def _add_latlon(df):
    """Add WGS84 latitude/longitude derived from the native x/y coords."""
    src_crs = df.attrs.get("crs", "EPSG:4326")
    if str(src_crs).upper().endswith("4326"):
        df["longitude"] = np.round(df["x"], config.COORD_DECIMALS)
        df["latitude"] = np.round(df["y"], config.COORD_DECIMALS)
    else:
        tf = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(df["x"].values, df["y"].values)
        df["longitude"] = np.round(lon, config.COORD_DECIMALS)
        df["latitude"] = np.round(lat, config.COORD_DECIMALS)
    return df


def _flag_orthomosaic_coverage(df):
    """Mark plants whose native x/y fall within the orthomosaic extent."""
    with rasterio.open(str(config.ORTHO_TIF)) as src:
        b = src.bounds
    inside = ((df["x"] >= b.left) & (df["x"] <= b.right) &
              (df["y"] >= b.bottom) & (df["y"] <= b.top))
    df["in_orthomosaic"] = inside
    outside = int((~inside).sum())
    if outside:
        print(f"[pipeline] {outside} plants fall OUTSIDE the orthomosaic extent "
              f"(flagged in_orthomosaic=False)")
    return df


def run():
    print("=" * 64)
    print("  PLANT HEALTH INDEX PIPELINE")
    print("=" * 64)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load plant centres from the ML team.
    plants = load_plants(config.PLANTS_FILE)

    # 2. Geographic coordinates for the output.
    plants = _add_latlon(plants)

    # 2b. Flag plants that lie outside the orthomosaic imagery.
    plants = _flag_orthomosaic_coverage(plants)

    # 2c. Physical canopy area from pixel area.
    if "area_px" in plants.columns:
        plants["canopy_area_m2"] = (plants["area_px"]
                                    * config.ORTHO_GSD_M ** 2).round(4)

    # 3. Cross-check each plant against the NDVI and NDRE rasters.
    have_masks = config.MASKS_DIR.exists() and "mask_file" in plants.columns
    use_raw = (config.USE_RAW_INDEX_CSV and config.RAW_INDEX_CSV.exists()
               and have_masks)
    if use_raw:
        from raw_index_sampler import assign_raw_canopy_indices
        for col, vals in assign_raw_canopy_indices(
                plants, config.ORTHO_TIF, config.MASKS_DIR, config.RAW_INDEX_CSV,
                plants_crs=config.PLANTS_CRS, max_match_m=config.RAW_MAX_MATCH_M,
                noise_label=config.NOISE_LABEL,
                ndvi_cmap=NDVI_CMAP, ndre_cmap=NDRE_CMAP).items():
            plants[col] = vals
    elif config.USE_CANOPY_MASKS and have_masks:
        specs = [(config.NDVI_TIF, NDVI_CMAP, "ndvi"),
                 (config.NDRE_TIF, NDRE_CMAP, "ndre")]
        osavi_spec = None
        if config.USE_OSAVI_SOIL_MASK and config.OSAVI_TIF.exists():
            osavi_spec = (config.OSAVI_TIF, OSAVI_CMAP, config.OSAVI_SOIL_THRESHOLD)
        for col, vals in assign_canopy_indices(
                plants, config.ORTHO_TIF, config.MASKS_DIR, specs,
                noise_label=config.NOISE_LABEL, osavi_spec=osavi_spec).items():
            plants[col] = vals
    else:
        for col, vals in assign_index(plants, config.NDVI_TIF, NDVI_CMAP, "ndvi").items():
            plants[col] = vals
        for col, vals in assign_index(plants, config.NDRE_TIF, NDRE_CMAP, "ndre").items():
            plants[col] = vals

    # 3b. Fill index gaps with XGBoost (predicted plants flagged).
    if config.USE_XGB_IMPUTE:
        from xgb_imputer import impute_indices
        plants = impute_indices(plants, noise_label=config.NOISE_LABEL)

    # 4. Derive a plant-health assessment from the indices.
    plants = classify_health(plants, noise_label=config.NOISE_LABEL)

    # 5. Assemble a tidy, ordered output table.
    id_col = "plant_id" if "plant_id" in plants.columns else plants.columns[0]
    preferred = [
        id_col, "sector_label", "latitude", "longitude", "x", "y",
        "in_orthomosaic", "pixel_x", "pixel_y", "area_px", "canopy_area_m2",
        "health_score", "health_status", "health_color",
        "health_diagnosis", "limiting_factor",
        "biomass_score", "nitrogen_score",
        "health_cluster", "cluster_confidence", "is_anomaly", "index_source",
        "ndvi", "ndvi_smoothed", "ndvi_category", "ndvi_match_dist", "ndvi_canopy_px",
        "ndre", "ndre_category", "ndre_match_dist", "ndre_canopy_px",
        "osavi", "canopy_soil_frac",
    ]
    ordered = [c for c in preferred if c in plants.columns]
    ordered += [c for c in plants.columns if c not in ordered]
    out = plants[ordered]

    # Surface plants that actually have index values first; NOISE / out-of-
    # coverage rows (blank ndvi/ndre) sink to the bottom.
    out = out.assign(_has=out["ndvi"].notna()).sort_values(
        ["_has", "sector_label", id_col],
        ascending=[False, True, True]).drop(columns="_has")

    out.to_csv(config.OUTPUT_CSV, index=False)
    print("-" * 64)
    print(f"  Saved {len(out):,} rows -> {config.OUTPUT_CSV}")
    print(f"  NDVI assigned : {out['ndvi'].notna().sum():,}")
    print(f"  NDRE assigned : {out['ndre'].notna().sum():,}")
    print(f"  Health scored : {out['health_score'].notna().sum():,}")
    if out["ndvi"].notna().any():
        print(f"  NDVI mean     : {out['ndvi'].mean():.3f}")
    if out["ndre"].notna().any():
        print(f"  NDRE mean     : {out['ndre'].mean():.3f}")
    print("=" * 64)
    return out


if __name__ == "__main__":
    run()
