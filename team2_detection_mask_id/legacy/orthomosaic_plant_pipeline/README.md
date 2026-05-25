# Orthomosaic Plant Pipeline

This folder is a standalone, GitHub-ready pipeline that takes a drone orthomosaic as input and produces separate outputs for plant masks, plant coordinates, plant IDs, sectors, and an HTML viewer whose background is the masked orthomosaic.

## What the pipeline does

The pipeline runs in five stages:

1. Tile the orthomosaic into overlapping windows.
2. Detect plants in each tile with YOLO.
3. Segment each detected plant with SAM and stitch all tile masks back into one orthomosaic-sized mask.
4. Convert the stitched mask into individual plant instances, assign sectors, rows, columns, and deterministic plant IDs.
5. Export separate files for masks, coordinates, IDs, GIS points, and a standalone HTML viewer.

## Main concepts used

### Tiling

Orthomosaics are usually much larger than the native input size of YOLO and SAM. The pipeline therefore cuts the raster into tiles, typically `640 x 640`, with overlap.

Why overlap is used:

- edge plants are less likely to be clipped
- detection quality is better near tile borders
- duplicate detections are filtered using each box center

### Detection and segmentation

YOLO predicts plant bounding boxes for each tile. Those boxes are passed to SAM as prompts. SAM returns masks, which are cleaned and stitched into one full-scene binary mask.

### Plant instances

After stitching, connected-component labeling is used to turn the binary mask into one instance per connected plant blob. That gives:

- one `instance_id` per plant
- centroid in pixel space
- centroid in orthomosaic coordinates
- area in pixels
- bounding box for cropped per-plant masks

### Sector assignment

The centroid cloud is clustered with HDBSCAN. Each cluster becomes a sector. The pipeline then estimates row and column order inside each sector using the sector geometry.

### Stable plant IDs

Each plant gets a deterministic ID:

```text
PLT-S02-R014-C007-A3F2C1
```

Meaning:

- `PLT`: fixed plant prefix
- `S02`: sector label
- `R014`: row index within the sector
- `C007`: column index within the row
- `A3F2C1`: short hash from the plant coordinates

## Folder structure

```text
orthomosaic_plant_pipeline/
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.cmd
├── run_pipeline.py
├── inputs/
├── outputs/
├── weights/
└── pipeline/
    ├── __init__.py
    ├── clustering.py
    ├── core.py
    ├── plant_ids.py
    └── viewer.py
```

## Inputs

### Required input 1: orthomosaic

A georeferenced `.tif` or `.tiff` orthomosaic.

Default location:

```text
inputs/your_orthomosaic.tif
```

### Required input 2: YOLO weights

A trained detector checkpoint.

Default location:

```text
weights/best.pt
```

### Required input 3: SAM weights

A SAM checkpoint.

Default location:

```text
weights/sam_b.pt
```

## Outputs

The outputs are intentionally separated by stage.

### `outputs/01_segmentation/`

- `orthomosaic_binary_mask.tif`: stitched binary mask of all plants
- `orthomosaic_binary_mask.json`: compact JSON export of the stitched binary mask using per-row x-runs
- `orthomosaic_overlap_count.tif`: count of how many tile masks touched each pixel
- `orthomosaic_overlap_count.json`: compact JSON export of overlap counts using per-row x-runs with values
- `orthomosaic_masked_overlay.tif`: orthomosaic with plant mask overlay
- `orthomosaic_masked_overlay_preview.png`: lightweight preview for the viewer
- `stitched_mask.dat`: temporary intermediate memmap used while stitching masks
- `stitched_mask_counts.dat`: temporary intermediate memmap used while stitching overlap counts

### `outputs/02_instances/`

- `plant_instance_labels.tif`: integer-labeled raster where each plant has its own label
- `plant_instances.csv`: table of extracted plant instances
- `plant_instances.json`: JSON version of the instance table, including per-plant coordinates and the cropped binary mask encoded as row runs
- `plant_instances_pre_id.json`: strict pre-ID JSON export written immediately after instance extraction, with coordinates, bbox, and mask for each plant instance

### `outputs/03_ids/`

- `plant_coordinates.csv`: dedicated coordinate export
- `plant_ids.csv`: dedicated ID and sector export
- `plants_with_ids.csv`: main spreadsheet-friendly output
- `plants_with_ids.json`: JSON version of the same plant records
- `plants_points.geojson`: GIS-ready points with coordinates and plant IDs

### `outputs/04_masks/`

- one PNG mask crop per plant, for example `PLT-S00-R001-C001-ABC123_mask.png`
- `mask_index.json`: map of plant IDs to mask filenames and bounding boxes

### `outputs/05_viewer/`

- `orthomosaic_masked_overlay_preview.png`: viewer background
- `viewer.html`: standalone HTML inspection viewer

### Root output

- `outputs/run_summary.json`: master summary with metadata and file locations

## Installation

```powershell
cd C:\drone_big\orthomosaic_plant_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

### Option A: use the default folder layout

1. Put the orthomosaic inside `inputs/`.
2. Put the detector weights inside `weights/best.pt`.
3. Put the SAM checkpoint inside `weights/sam_b.pt`.
4. Run:

```powershell
python .\run_pipeline.py
```

### Option B: pass explicit paths

```powershell
python .\run_pipeline.py `
  --source C:\path\to\orthomosaic.tif `
  --yolo-weights C:\path\to\best.pt `
  --sam-weights C:\path\to\sam_b.pt `
  --output-dir C:\path\to\outputs
```

## Command reference

```powershell
python .\run_pipeline.py `
  --source C:\path\to\orthomosaic.tif `
  --yolo-weights C:\path\to\best.pt `
  --sam-weights C:\path\to\sam_b.pt `
  --output-dir C:\path\to\outputs `
  --tile-size 640 `
  --overlap 0.10 `
  --conf 0.25 `
  --imgsz 640 `
  --device 0 `
  --min-mask-area 50 `
  --min-instance-area 80 `
  --min-cluster-size 25 `
  --preview-max-dim 4096
```

## Parameter guide

- `--source`: orthomosaic path. If omitted, the pipeline auto-selects one TIFF from `inputs/`.
- `--yolo-weights`: detector checkpoint path.
- `--sam-weights`: SAM checkpoint path.
- `--output-dir`: output root directory.
- `--tile-size`: tile width and height in pixels.
- `--overlap`: tile overlap fraction.
- `--conf`: YOLO confidence threshold.
- `--imgsz`: YOLO inference image size.
- `--device`: GPU id like `0` or `cpu`.
- `--prefetch-workers`: background threads used to preload and preprocess upcoming tiles during segmentation.
- `--min-mask-area`: minimum pixel area kept from SAM masks before stitching.
- `--min-instance-area`: minimum connected-component area accepted as a plant.
- `--min-cluster-size`: HDBSCAN minimum cluster size used for sector creation.
- `--preview-max-dim`: maximum size of the viewer preview image.

## How coordinates are generated

The pipeline exports:

- `pixel_x`, `pixel_y`: centroid in image pixels
- `geo_x`, `geo_y`: centroid transformed through the orthomosaic geotransform

So the geographic coordinates are correct only if the orthomosaic itself is correctly georeferenced.

## How masks are generated

The pipeline writes three useful mask forms:

1. `orthomosaic_binary_mask.tif`
   One full-scene binary vegetation mask.

2. `orthomosaic_binary_mask.json`
   A compact JSON version of the stitched mask, stored as sparse row runs instead of one value per pixel.

3. `plant_instance_labels.tif`
   One full-scene integer raster where every plant has a unique label.

4. `outputs/04_masks/*.png`
   One cropped mask per plant for QA, annotation, or downstream ML tasks.

## Recommended usage workflow

1. Prepare the trained detection weights.
2. Add the SAM checkpoint.
3. Add the orthomosaic.
4. Run the pipeline.
5. Check `outputs/run_summary.json`.
6. Open `outputs/05_viewer/viewer.html`.
7. Use `plants_with_ids.csv` and `outputs/04_masks/` in downstream analysis.

## Troubleshooting

### No orthomosaic found

Place exactly one TIFF in `inputs/`, or pass `--source`.

### Multiple orthomosaics found

Pass `--source` explicitly.

### Viewer looks empty

Check:

- whether any plants were detected
- whether the search box is filtering results
- whether `plants_with_ids.csv` contains rows

### Too many false positives

Try:

- increasing `--conf`
- increasing `--min-mask-area`
- increasing `--min-instance-area`

### Sector grouping looks unstable

Try increasing `--min-cluster-size`.

## Assumptions

- the orthomosaic is georeferenced correctly
- the YOLO model is trained for the target crop
- one connected segmented blob approximately corresponds to one plant
- sectors are inferred from centroid clustering, not from manually supplied farm-sector boundaries

## GitHub notes

This folder is prepared to be pushed as its own clean project slice:

- code and docs are tracked
- model weights are ignored
- local orthomosaic inputs are ignored
- generated outputs are ignored

Typical next commands:

```powershell
git add orthomosaic_plant_pipeline
git commit -m "Add standalone orthomosaic plant pipeline"
```
