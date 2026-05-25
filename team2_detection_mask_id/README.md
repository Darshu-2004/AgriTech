# Pineapple Detection Pipeline

End-to-end pipeline for detecting and mapping pineapple plants from drone orthomosaics using YOLOv8 + SAM segmentation and HDBSCAN-based plant ID assignment.

## Quick Start

```bash
pip install -r requirements.txt

# Run the full pipeline (auto-detects TIFF in ortho/)
python run_pipeline.py

# Explicit paths
python run_pipeline.py --ortho ortho/ggs-orthophoto.tif --output-dir outputs/my_run

# Resume after Stage 1 already ran
python run_pipeline.py --skip-ortho --output-dir outputs/my_run
```

Results land in `outputs/run_TIMESTAMP/` (or your `--output-dir`):
```
outputs/run_20260525_143000/
├── 01_orthomosaic/
│   ├── orthomosaic_mask.tif          # binary mask GeoTIFF
│   ├── orthomosaic_overlay.tif       # colour overlay GeoTIFF
│   ├── orthomosaic_overlay_preview.png
│   └── run_summary.json              # tile-by-tile detection log
└── 02_plant_ids/
    ├── plants.geojson                # all plants with IDs + sector labels
    ├── plant_id_summary.json
    ├── farm_boundary.geojson
    └── viewer.html                   # open in browser
```

## Pipeline Stages

| Stage | Folder | Description |
|-------|--------|-------------|
| 1 | `01_dataset_prep/` | Split labelled images into train/val/test dataset |
| 2 | `02_training/` | Train YOLOv8 (s/m/l/x) on the prepared dataset |
| 3 | `03_evaluation/` | Evaluate trained models, generate comparison report |
| 4 | `04_orthomosaic/` | Tile orthomosaic → YOLO detect → SAM segment → stitch mask |
| 5 | `05_plant_ids/` | Cluster mask centroids → assign row/column plant IDs |

## Requirements

```
Python 3.10+
CUDA GPU recommended (falls back to CPU with --device cpu)
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Trained Models

Best model: **YOLOv8x** trained for 200 epochs.

| Model | mAP50-95 | Precision | Recall |
|-------|----------|-----------|--------|
| yolov8s | 0.8365 | 0.9509 | 0.9854 |
| yolov8m | 0.8475 | 0.9546 | 0.9763 |
| yolov8l | 0.8333 | 0.9609 | 0.9682 |
| yolov8x | 0.8370 | 0.9540 | 0.9764 |

Full training report: [`reports/FULL_MODEL_TRAINING_REPORT.md`](reports/FULL_MODEL_TRAINING_REPORT.md)

## File Layout

```
drone_big/
├── run_pipeline.py          # Master command
├── requirements.txt
├── 01_dataset_prep/         # Dataset preparation
├── 02_training/             # Model training
├── 03_evaluation/           # Model evaluation & reports
├── 04_orthomosaic/          # Orthomosaic tiling + detection + segmentation
├── 05_plant_ids/            # Plant clustering + ID assignment
├── dataset/                 # YOLO labels + data.yaml (images not tracked)
├── reports/                 # Training reports and docs
└── legacy/                  # Earlier pipeline versions
```
