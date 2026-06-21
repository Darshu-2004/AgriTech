# Plant Health Index Pipeline

Computes per-plant **NDVI** and **NDRE** canopy means from drone imagery and
derives a plant-health assessment, then renders an interactive map. Built for
the GGS plantation dataset.

```
plant detections + canopy masks  +  NDVI/NDRE maps  +  orthomosaic
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

Running `run.py` executes every stage end-to-end and writes
`output/plant_health_indices.csv`.

## Inputs (all paths set in `config.py`)

| Input | File | Role |
|-------|------|------|
| Plant table | `../outputs 2/03_ids/plants_with_ids.csv` | plant IDs, geo coords, bboxes, mask names |
| Canopy masks | `../outputs 2/04_masks/*.png` | one binary mask per plant (sized to its bbox) |
| NDVI map | `../maps/Task-...-NDVI.tif` | colorized index raster (index = filename suffix) |
| NDRE map | `../maps/Task-...-NDRE.tif` | colorized index raster |
| Orthomosaic | `../ggs-orthophoto (2).tif` | basemap + reference grid for the masks |

Notes:
- The plant detections were generated from `ggs-orthophoto (2).tif`, so plant
  coordinates register to it **exactly** — the map overlay is pixel-perfect.
- An OSAVI map exists in `maps/` but is intentionally **not used**.
- The NDVI/NDRE maps are colorized RGBA renderings; values are recovered by
  matching each pixel to the nearest colormap bin (the GGS/`NDVI_gene.py`
  palette) and taking the bin value.

## How a value is computed

For every plant, its binary canopy mask is projected from the ggs-ortho pixel
grid into each index raster via a composed affine transform, and the index is
**averaged over the plant's actual canopy pixels** — not a fixed window. This
is the most accurate per-plant value available from these inputs.

Then the health engine:

1. **Spatial smoothing** — each plant's NDVI is blended with its 8 nearest
   neighbours (KD-Tree) to suppress noise.
2. **Vigor score** — NDVI + NDRE combined and scaled to *this dataset's*
   distribution (robust p10–p90 percentiles), so the spread is meaningful
   regardless of absolute index scale.
3. **Anomaly blend** — an IsolationForest spectral-outlier score is mixed in.
4. **Classification** — a 0–100 `health_score` binned into:

   | Status | Score | Color |
   |--------|-------|-------|
   | Healthy | 70–100 | `#2E7D32` |
   | Moderate | 40–70 | `#FBC02D` |
   | Stressed | 15–40 | `#FF5722` |
   | Critical | 0–15 | `#D32F2F` |
   | Out of Boundary | no index | `#1976D2` |
   | NOISE | weed / false positive | `#757575` |

`NOISE` detections and plants outside the index coverage get blank index +
health values.

## Modules

| File | Role |
|------|------|
| `config.py` | All paths, source toggles & constants |
| `plant_loader.py` | Read the plant table (CSV or GeoJSON) → DataFrame |
| `colormaps.py` | NDVI/NDRE palettes + reverse RGB→value lookup |
| `canopy_sampler.py` | **Default sampler** — canopy mean from the colorized maps |
| `raw_index_sampler.py` | Optional sampler — canopy mean from continuous CSV values |
| `raster_sampler.py` | Window / Otsu-shadow sampling (fallback, no masks) |
| `index_assigner.py` | Per-point sampling wrapper (fallback path) |
| `health_classifier.py` | Spatial smoothing + vigor + anomaly + health tiers |
| `pipeline.py` | Orchestrates all stages, writes the CSV |
| `make_visualization.py` | Build the interactive HTML map |
| `run.py` | Entry point |

## Sampling source (configurable in `config.py`)

The pipeline picks a source in this order:

1. **Raw continuous CSV** (`USE_RAW_INDEX_CSV`) — real NDVI/NDRE values, but
   only ~76% field coverage. Currently **off**.
2. **Canopy masks over the colorized maps** (`USE_CANOPY_MASKS`) — ~96%
   coverage. **Active default.**
3. **Window / shadow-mask sampling** — used only when no canopy masks exist.

## Output CSV columns

`plant_id, sector_label, latitude, longitude, x, y, in_orthomosaic,
pixel_x, pixel_y, area_px, canopy_area_m2, health_score, health_status,
health_color, ndvi, ndvi_smoothed, ndvi_category, ndvi_match_dist,
ndvi_canopy_px, ndre, ndre_category, ndre_match_dist, ndre_canopy_px, …`
(plus the original ML columns: instance_id, sector_id, bbox_*, geo_*, mask_file).

Key fields:
- `ndvi`, `ndre` — **canopy-mean index value** for the plant
- `ndvi_canopy_px` / `ndre_canopy_px` — number of canopy pixels averaged
- `ndvi_category` / `ndre_category` — human-readable index class
- `canopy_area_m2` — physical canopy size (`area_px × GSD²`)
- `health_score` / `health_status` / `health_color`

Rows are sorted so plants **with** index values appear first; blank
(NOISE / out-of-coverage) rows sink to the bottom.

## Visualization

`make_visualization.py` builds `output/plant_health_map.html` — an interactive
Leaflet map over the orthomosaic, using `L.CRS.Simple` in UTM space so markers
align exactly with the imagery, and a canvas renderer for smooth panning with
~14.5k points. Color by **Health status** (default), NDVI or NDRE; filter by
sector; click any plant for its full record. Align X/Y nudge sliders exist for
fine registration (default 0; not needed with the matched ortho).

## Requirements

```bash
pip install -r requirements.txt
# numpy, pandas, rasterio, pyproj, pillow, scikit-learn, scipy
```
