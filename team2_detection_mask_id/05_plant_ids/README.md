# Stage 5 — Plant ID Assignment

Reads the stitched mask from Stage 4, extracts plant centroids, clusters them into field sectors with HDBSCAN, assigns deterministic row/column IDs, and exports GeoJSON + an interactive HTML viewer.

## Usage

```bash
python 05_plant_ids/run_plant_id_pipeline.py \
    --source ortho/ggs-orthophoto.tif \
    --run-summary pipeline_outputs/orthomosaic_full/run_summary.json \
    --mask pipeline_outputs/orthomosaic_full/orthomosaic_mask.tif \
    --output-dir pipeline_outputs/plant_ids
```

Or use the master command which runs both stages automatically:
```bash
python run_pipeline.py
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | required | Path to original orthomosaic GeoTIFF |
| `--run-summary` | required | Path to `run_summary.json` from Stage 4 |
| `--mask` | optional | Path to `orthomosaic_mask.tif` (recommended) |
| `--boundary` | optional | GeoJSON farm boundary; auto-detects via NDVI if omitted |
| `--output-dir` | `pipeline_outputs/plant_ids` | Output directory |
| `--min-cluster-size` | `25` | HDBSCAN min cluster size (increase for larger farms) |

## How it works

1. Extracts one centroid per connected blob from the mask raster
2. Optionally filters centroids to a farm boundary polygon
3. Clusters centroids with HDBSCAN into field sectors
4. Rescues edge/noise points using sector geometry (PCA axes)
5. Within each sector, assigns row and column indices by orientation
6. Builds deterministic plant IDs in the form `PLT-S02-R014-C007`
7. Exports GeoJSON, summary JSON, and an interactive Leaflet viewer

## Plant ID format

```
PLT-S02-R014-C007-A3F2C1
     |    |    |    |
     |    |    |    +-- short hash from geo coordinates
     |    |    +------- column index within the sector row
     |    +------------ row index within the sector
     +----------------- sector label (HDBSCAN cluster)
```

## Output

```
output_dir/
├── plants.geojson              # all plants with IDs, sector, row, col
├── plant_id_summary.json       # metadata + per-sector counts
├── farm_boundary.geojson
├── exclusion_mask_outline.geojson
└── viewer.html                 # open in any browser — no server needed
```

## Previous run results (ggs-orthophoto.tif)

| Metric | Value |
|--------|-------|
| Plant centroids extracted | 14,797 |
| Noise / unclustered | 402 (2.7%) |
| Sectors identified | 14 (S00–S13) |
| Largest sector (S00) | 3,299 plants |
