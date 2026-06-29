# 🍍 Unsupervised Crop Health & Scoring Pipeline (Unified Team Edition)

This repository contains the enterprise-grade precision agriculture analytics system designed to ingest crop coordinates, extract georeferenced vegetation indices, predict growth stages, estimate vegetative biomass, and calculate continuous crop health scores using cohort-based anomaly detection.

---

## 🗺️ Pipeline Flowchart

```text
                      RGB Orthomosaic
                             +
                 NDRE, OSAVI, NDVI Rasters
                             +
                     Plants Dataset CSV
               (plant_id, geo_x, geo_y, area_px,
                 bbox, sector_label, etc.)
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │        01_extract_indices.py (Step 1)        │
        │  [Parallel Index Sampling Engine]            │
        │  • Split coordinates into CPU-core chunks     │
        │  • Execute parallel worker queries in RAM    │
        │                                              │
        │  [OSAVI Soil-Gating & Shadow Masking]        │
        │  • Map native coordinates using affine transforms│
        │  • Filter out pixels with OSAVI < 0.2        │
        │  • Extract indices from sunlit leaves only   │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
               plants_extracted_indices.csv
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │        train_growth_model.py (Step 1b)       │
        │  • Read GSD tags from GeoTIFF via PIL        │
        │  • Convert area to physical canopy_area (m²) │
        │  • Train RandomForestClassifier              │
        │  • Save: growth_stage_rf.pkl                 │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │         03_classify_health.py (Step 3)       │
        │  [XGBoost Index Gap-Imputer]                 │
        │  • Fill out-of-boundary crops via k-NN ML    │
        │                                              │
        │  [Stage-Aware Inference]                     │
        │  • Load RF model & predict physical stages   │
        │                                              │
        │  [KD-Tree Spatial Smoothing]                 │
        │  • Blend values with K=8 spatial neighbors   │
        │                                              │
        │  [Unsupervised Anomaly Detection]            │
        │  • Fit IsolationForest per growth stage cohort│
        │                                              │
        │  [Biomass & Yield Estimates]                 │
        │  • Calculate crop biomass (kg) & sector (t)   │
        │                                              │
        │  [Health Score Engine (0-100)]               │
        │  • 70–100 → Healthy    (#2E7D32, Code 0)     │
        │  • 40–70  → Moderate   (#FBC02D, Code 1)     │
        │  • 15–40  → Stressed   (#FF5722, Code 2)     │
        │  • 0–15   → Critical   (#D32F2F, Code 3)     │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
            plants_with_predictions_health.csv
            sector_biomass_predictions.csv
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │        02_generate_preview.py (Step 2)       │
        │  • Generate plantation overlay Leaflet.js map│
        │  • Include interactive popups & legends      │
        │  • Save: plants_health_map.html              │
        └──────────────────────────────────────────────┘
```

---

## 🛠️ Requirements & Installation

The pipeline runs on Python 3.8+ and requires standard scientific Python libraries along with `rasterio` for coordinate parsing.

To install the required Python packages, run:
```bash
pip install numpy pandas pillow scikit-learn scipy xgboost rasterio psycopg2
```

---

## 🚀 How to Run the Code

### Step 1: Ingest Inputs
Make sure files are located inside:
* `dataset1/`: GeoTIFFs (OSAVI, NDVI, NDRE, and true-color orthomosaic).
* `outputs/03_ids/plants_with_ids.csv`: Crop boundary coordinates.
* `outputs/04_masks/`: Individual binary png canopy masks.

### Step 2: Execute Master Orchestrator
To run all pipeline stages sequentially, run:
```bash
python scripts/run_pipeline.py
```

---

## 💾 Outputs Generated

1. **`plants_with_predictions_health.csv`** (Workspace Root)
   - The final processed crop health database matching the PostGIS schema (noise crops filtered out).
   - Columns: `plant_id`, `sector_id`, `sector_label`, `pixel_x`, `pixel_y`, `geo_x`, `geo_y`, `area_px`, `canopy_area`, `predicted_growth_stage`, `health_score`, `health_status`, `health_status_code`, `health_color`, `osavi`, `ndvi`, `ndre`, and empty telemetry placeholders.

2. **`sector_biomass_predictions.csv`** (Workspace Root)
   - Aggregated 6-column database: `sector_id`, `sector_label`, `total_active_plants`, `predicted_biomass_tonnes`, `predicted_yield_tonnes` (blank), `market_grade` (blank).

3. **`plants_health_map.html`** (Workspace Root)
   - An interactive Leaflet.js HTML map overlaying crop circles (color-coded by health status code) on top of the downsampled web orthomosaic background.
