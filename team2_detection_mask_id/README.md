# Pineapple Detection Pipeline

End-to-end pipeline for detecting and mapping pineapple plants from drone orthomosaics using YOLOv8 + SAM segmentation and HDBSCAN-based plant ID assignment.

## Setup on a New Machine

### 1. Clone the repo
```bash
git clone https://github.com/Darshu-2004/AgriTech.git
cd AgriTech/team2_detection_mask_id
```

### 2. Install Git LFS and pull model weights
```bash
git lfs install
git lfs pull
```
This downloads the 817 MB of model weights stored via Git LFS.

### 3. Move weights to expected locations
```bash
# Windows
copy weights\sam_b.pt sam_b.pt
mkdir runs\train\pineapple_yolov8x2\weights
copy weights\yolov8x2_best.pt runs\train\pineapple_yolov8x2\weights\best.pt

# Linux / Mac
cp weights/sam_b.pt sam_b.pt
mkdir -p runs/train/pineapple_yolov8x2/weights
cp weights/yolov8x2_best.pt runs/train/pineapple_yolov8x2/weights/best.pt
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```
Requires **Python 3.10+**. A **CUDA GPU is recommended** (use `--device cpu` for CPU-only, much slower).

### 5. Add your orthomosaic
Drop your GeoTIFF drone orthomosaic into the `ortho/` folder:
```
ortho/your_farm.tif
```

### 6. Run the pipeline
```bash
# Auto-detects TIFF in ortho/
python run_pipeline.py

# Explicit paths and output folder
python run_pipeline.py --ortho ortho/your_farm.tif --output-dir outputs/my_run

# CPU only
python run_pipeline.py --device cpu

# Skip Stage 1 if mask already exists
python run_pipeline.py --skip-ortho --output-dir outputs/my_run
```

## Output

Results land in `outputs/run_TIMESTAMP/` (or your `--output-dir`):
```
outputs/run_20260525_143000/
├── 01_orthomosaic/
│   ├── orthomosaic_mask.tif           # binary mask GeoTIFF
│   ├── orthomosaic_overlay.tif        # colour overlay GeoTIFF
│   ├── orthomosaic_overlay_preview.png
│   └── run_summary.json               # tile-by-tile detection log
└── 02_plant_ids/
    ├── plants.geojson                 # all plants with IDs + sector labels
    ├── plant_id_summary.json
    ├── farm_boundary.geojson
    └── viewer.html                    # open in any browser — no server needed
```

## What's in this repo

| Stage | Folder | Description |
|-------|--------|-------------|
| 1 | `01_dataset_prep/` | Split labelled images into train/val/test dataset |
| 2 | `02_training/` | Train YOLOv8 (s/m/l/x) on the prepared dataset |
| 3 | `03_evaluation/` | Evaluate trained models, generate comparison report |
| 4 | `04_orthomosaic/` | Tile orthomosaic → YOLO detect → SAM segment → stitch mask |
| 5 | `05_plant_ids/` | Cluster mask centroids → assign row/column plant IDs |

## What is NOT in this repo (too large)

| Item | Why excluded | What to do |
|------|-------------|------------|
| `ortho/*.tif` | 2+ GB input file | Provide your own GeoTIFF |
| `dataset/images/` | Hundreds of MB | Only labels are tracked |
| `runs/` training logs | Large plots/CSVs | Only `best.pt` weight is included |
| `outputs/` | Generated per-run | Created automatically on first run |

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
team2_detection_mask_id/
├── run_pipeline.py          # Master command — run this
├── requirements.txt
├── weights/                 # Model weights (Git LFS)
│   ├── sam_b.pt
│   ├── yolov8x2_best.pt     # trained best checkpoint
│   ├── yolov8x.pt
│   ├── yolov8l.pt
│   ├── yolov8m.pt
│   └── yolov8s.pt
├── ortho/                   # Drop your GeoTIFF here (not tracked)
├── 01_dataset_prep/         # Dataset preparation
├── 02_training/             # Model training
├── 03_evaluation/           # Model evaluation & reports
├── 04_orthomosaic/          # Orthomosaic tiling + detection + segmentation
├── 05_plant_ids/            # Plant clustering + ID assignment
├── dataset/                 # YOLO labels + data.yaml (images not tracked)
├── reports/                 # Training reports and docs
└── legacy/                  # Earlier pipeline versions
```
