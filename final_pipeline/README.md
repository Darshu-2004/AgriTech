# Plant Health Pipeline

End-to-end plant-health analytics for the GGS plantation: it computes per-plant
**NDVI** and **NDRE** canopy means from drone imagery, removes soil noise with
**OSAVI**, fills gaps with **XGBoost**, derives an agronomic health model, and
renders an interactive map.

```
plant detections + canopy masks
NDVI / NDRE / OSAVI maps
true-color orthomosaic
            │
            ▼
  output/plant_health_indices.csv     (one row per plant)
  output/plant_health_map.html        (interactive map)
```

## Run

```bash
cd final_pipeline
../.venv/bin/python run.py                 # build the CSV
../.venv/bin/python make_visualization.py  # build the HTML map
```

`run.py` executes every stage end-to-end.

## Inputs (all paths in `config.py`)

| Input | File | Role |
|-------|------|------|
| Plant table | `../outputs 2/03_ids/plants_with_ids.csv` | IDs, geo coords, bboxes, mask names |
| Canopy masks | `../outputs 2/04_masks/*.png` | one binary mask per plant (sized to bbox) |
| NDVI map | `../maps/Task-...-NDVI.tif` | colorized index raster (biomass) |
| NDRE map | `../maps/Task-...-NDRE.tif` | colorized index raster (nitrogen) |
| OSAVI map | `../maps/Task-...-OSAVI.tif` | soil-adjusted index — **used as a soil gate** |
| Orthomosaic | `../ggs-orthophoto (2).tif` | basemap + reference grid for the masks |

Key facts:
- The plant detections were generated from `ggs-orthophoto (2).tif`, so plant
  coordinates register to it **exactly** — the map overlay is pixel-perfect.
- The NDVI/NDRE/OSAVI maps are colorized RGBA renderings; values are recovered
  by matching each pixel to the nearest colormap bin (the GGS RdYlGn palette).
- The index is identified by the **filename suffix** (NDVI / NDRE / OSAVI).

## How a plant's value is computed

1. **Canopy sampling** — each plant's binary mask is projected from the
   ggs-ortho pixel grid into each index raster via a composed affine transform.
2. **OSAVI soil gate** — for every canopy pixel, its OSAVI value is checked;
   pixels below `OSAVI_SOIL_THRESHOLD` (0.2) are **bare soil and discarded**, so
   NDVI/NDRE are averaged over **vegetation pixels only**. This removes the
   soil/background noise that otherwise contaminates sparse or open canopies.
   *(On this dataset ~62% of raw mask pixels were soil; gating lifted the means
   from NDVI 0.43→0.57, NDRE −0.06→0.08.)*
3. **XGBoost gap-filling** — plants the maps don't cover (or whose canopy was
   entirely soil) get NDVI/NDRE predicted from canopy morphology + position +
   spatial neighbours, flagged `index_source = 'predicted'`
   (hold-out R² ≈ 0.82–0.87).
4. **Health model** — see below.

`NOISE` detections (weeds / false positives) are always left blank.

## Health model (`health_classifier.py`)

Two indices, two biological meanings — kept as separate axes:

- **NDVI → biomass** (leaf area / canopy density) → `biomass_score` (0–100)
- **NDRE → nitrogen** (chlorophyll / N uptake) → `nitrogen_score` (0–100)

Both scores are the plant's **percentile rank within the field**. Stages:

1. KD-Tree spatial smoothing of each index (8 neighbours).
2. `biomass_score` & `nitrogen_score` (percentile ranks).
3. **GaussianMixture** over (NDVI, NDRE) → data-driven plant archetypes
   (`health_cluster` + `cluster_confidence`).
4. **IsolationForest** spectral-outlier flag (`is_anomaly`).
5. `health_score = 0.55·biomass + 0.45·nitrogen` → 4-tier `health_status`
   (Healthy / Moderate / Stressed / Critical), plus an interpretable
   `health_diagnosis` (e.g. "Nitrogen deficient") and `limiting_factor`
   (Nitrogen / Biomass / Balanced).

## Modules

| File | Role |
|------|------|
| `config.py` | All paths, source toggles & thresholds |
| `plant_loader.py` | Read the plant table (CSV or GeoJSON) |
| `colormaps.py` | NDVI / NDRE / OSAVI palettes + reverse RGB→value lookup |
| `canopy_sampler.py` | **Default** — canopy mean from colorized maps + OSAVI soil gate |
| `raw_index_sampler.py` | Optional — canopy mean from the continuous NDVI/NDRE CSV |
| `raster_sampler.py` | Window / Otsu-shadow sampling (fallback, no masks) |
| `index_assigner.py` | Per-point sampling wrapper (fallback path) |
| `xgb_imputer.py` | XGBoost NDVI/NDRE gap-filling |
| `health_classifier.py` | Smoothing + biomass/nitrogen scores + GMM + health tiers |
| `pipeline.py` | Orchestrates all stages, writes the CSV |
| `make_visualization.py` | Build the interactive HTML map |
| `run.py` | Entry point |

## Index source (configurable in `config.py`)

Chosen in this order:

1. **Raw continuous CSV** (`USE_RAW_INDEX_CSV`) — real NDVI/NDRE values, ~76%
   coverage. Currently **off**.
2. **Canopy masks over the colorized maps + OSAVI gate** (`USE_CANOPY_MASKS`,
   `USE_OSAVI_SOIL_MASK`) — **active default**.
3. **Window / shadow-mask sampling** — only when no canopy masks exist.

`USE_XGB_IMPUTE` then fills any remaining gaps.

## Output CSV — `output/plant_health_indices.csv`

One row per plant (rows with index values sorted first). Key columns:

| Column | Meaning |
|--------|---------|
| `ndvi`, `ndre` | **soil-free canopy mean** index values |
| `osavi` | mean OSAVI over the kept vegetation pixels |
| `canopy_soil_frac` | fraction of the mask that was soil (quality flag) |
| `ndvi_canopy_px` | vegetation pixels averaged |
| `biomass_score`, `nitrogen_score` | 0–100 field-relative agronomic scores |
| `health_score`, `health_status`, `health_color` | overall health tier |
| `health_diagnosis`, `limiting_factor` | actionable interpretation |
| `health_cluster`, `cluster_confidence`, `is_anomaly` | unsupervised ML outputs |
| `index_source` | `measured` or `predicted` (XGBoost) |
| `canopy_area_m2` | physical canopy size (`area_px × GSD²`) |

(plus the original ML columns: `instance_id`, `sector_*`, `bbox_*`, `geo_*`,
`mask_file`, `in_orthomosaic`, …)

## Visualization — `output/plant_health_map.html`

Interactive Leaflet map over the orthomosaic, using `L.CRS.Simple` in UTM space
(pixel-perfect alignment) and a canvas renderer (smooth with ~14.5k points).

- **Color by**: Biomass (NDVI), Nitrogen (NDRE), NDVI raw, NDRE raw.
- Filter by sector; click any plant for its NDVI/NDRE + biomass/nitrogen scores.
- Align X/Y nudge sliders for fine registration (default 0; not needed here).

Health status is computed and stored in the CSV but intentionally **not** shown
as a map layer.

## Requirements

```bash
pip install -r requirements.txt
# numpy, pandas, rasterio, pyproj, pillow, scikit-learn, scipy, xgboost
```
