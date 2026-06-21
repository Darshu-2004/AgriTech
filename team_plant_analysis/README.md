# 🍍 Unsupervised Crop Health & Scoring Pipeline (Pillow + Multiprocessing Edition)

This pipeline is an enterprise-grade precision agriculture analytics system designed to ingest crop coordinates, extract georeferenced vegetation indices, predict growth stages, and calculate continuous crop health scores using unsupervised anomaly detection.

By replacing C-dependent spatial libraries (`rasterio` and `tifffile`) with **Pillow (PIL)** and **NumPy**, this pipeline has **zero binary GDAL C++ dependencies**, making it extremely lightweight and easy to deploy in serverless environments, docker containers, or lightweight servers.

---

## 🗺️ Pipeline Flowchart (Phase 1)

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
        │  [Pillow-Based Multiprocessing Engine]       │
        │  • Split coordinates into CPU-core chunks     │
        │  • Execute parallel worker queries in RAM    │
        │                                              │
        │  [Otsu's Adaptive Shadow Masking]            │
        │  • Crop 10x10 RGB window dynamically         │
        │  • Apply pure-numpy Otsu's thresholding      │
        │  • Extract indices from sunlit leaves only   │
        │                                              │
        │  [Boundary Check]                            │
        │  • Flag Out-of-Boundary crops as NaN         │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
              plants_extracted_indices.csv
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │        train_growth_model.py (Step 1b)       │
        │  • Read GSD tags from GeoTIFF via PIL        │
        │  • Convert area to physical canopy_area_m²   │
        │  • Train RandomForestClassifier              │
        │  • Save: growth_stage_rf.pkl                 │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │         03_classify_health.py (Step 3)       │
        │  [Stage-Aware Inference]                     │
        │  • Load RF model & predict physical stages   │
        │                                              │
        │  [KD-Tree Spatial Smoothing]                 │
        │  • Blend values with K=8 spatial neighbors   │
        │                                              │
        │  [Dynamic Isolation Forest Anomaly]          │
        │  • Fit IsolationForest per growth stage      │
        │  • Score = Median - (1.5 * StdDev)           │
        │                                              │
        │  [Directional Outlier Vigor Capping]         │
        │  • Outliers with high OSAVI -> Healthy       │
        │  • Outliers with low OSAVI -> Unhealthy      │
        │                                              │
        │  [Health Score Engine (0-100)]               │
        │  • Score = f(NDRE%, OSAVI%, Stage-Area%)     │
        │  • Enforce absolute biological floors        │
        │                                              │
        │  • 70–100 → Healthy    (#2E7D32)             │
        │  • 40–70  → Moderate   (#FBC02D)             │
        │  • 15–40  → Stressed   (#FF5722)             │
        │  • 0–15   → Critical   (#D32F2F)             │
        │  • NaN    → Out of Bnd (#1976D2)             │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
                 plants_with_predictions_health.csv
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │        02_generate_preview.py (Step 2)       │
        │  • Generate plantation overlay preview       │
        │  • Include 4-tier health legend              │
        │  • Save: plants_health_map.png               │
        └──────────────────────────────────────────────┘
```

---

## 🛠️ Requirements & Installation

The pipeline runs on Python 3.8+ and does **not** require any binary GIS installations.

To install the required Python packages, run:
```bash
pip install numpy pandas pillow scikit-learn scipy matplotlib
```

---

## 🚀 How to Run the Code

### Step 1: Place Input Files
Make sure the following directory structure is set up:
```text
workspace_root/
  ├─ dataset1/
  │    ├─ Task-of-2026-03-22T063838646Z-orthophoto-OSAVI.tif
  │    ├─ Task-of-2026-03-22T070712342Z-orthophoto-NDVI.tif
  │    ├─ Task-of-2026-03-22T070712342Z-orthophoto-NDRE.tif
  │    └─ ggs-orthophoto (2).tif
  ├─ outputs/
  │    └─ 03_ids/
  │         └─ plants_with_ids.csv (contains crop coordinates & areas)
  └─ pipeline/
       ├─ 01_extract_indices.py
       ├─ train_growth_model.py
       ├─ 03_classify_health.py
       ├─ 02_generate_preview.py
       └─ run_pipeline.py
```

### Step 2: Execute the Master Orchestrator
To run all stages of the pipeline sequentially (Extraction $\rightarrow$ Training $\rightarrow$ Health Assessment $\rightarrow$ Visual Preview), run:
```bash
python pipeline/run_pipeline.py
```

---

## 💾 Outputs Generated

1. **`plants_with_predictions_health.csv`** (Workspace Root)
   - The final processed crop health database containing **25 columns** including:
     - `canopy_area_m2`: The GSD-scaled physical area.
     - `predicted_growth_stage_name`: Growth stage predicted by Random Forest.
     - `health_score`: Continuous health score ($0 - 100$).
     - `health_status`: The 4-tier health tier (`Healthy`, `Moderate`, `Stressed`, `Critical`, `Out of Boundary`, `NOISE`).
     - `health_color`: The hex code colors for database/GIS integration.
     - `osavi_smoothed`: Spatially smoothed OSAVI.

2. **`plants_health_map.png`** (Workspace Root)
   - A high-resolution plantation health visualization map displaying all individual crops color-coded by their health tier overlayed on the downscaled RGB background orthophoto.
