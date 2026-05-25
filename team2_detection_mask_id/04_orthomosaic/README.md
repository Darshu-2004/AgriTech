# Stage 4 — Orthomosaic Processing

Tiles a large GeoTIFF orthomosaic, runs YOLO detection and SAM segmentation on each tile, then stitches the masks back into a full-resolution georeferenced output.

## Usage

```bash
# Auto-detects TIFF in ortho/
python 04_orthomosaic/run_full_orthomosaic_pipeline.py

# Explicit source and output
python 04_orthomosaic/run_full_orthomosaic_pipeline.py \
    --source ortho/ggs-orthophoto.tif \
    --output-dir pipeline_outputs/my_run \
    --device 0

# Run segment pipeline on individual images (with polygon export)
python 04_orthomosaic/segment_pineapples_pipeline.py \
    --source dataset/images/test \
    --sam-weights sam_b.pt \
    --output-dir pipeline_outputs/test_set
```

## Arguments (run_full_orthomosaic_pipeline.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | auto | Path to orthomosaic GeoTIFF |
| `--output-dir` | `pipeline_outputs/orthomosaic_full` | Output directory |
| `--device` | `0` | GPU device or `cpu` |
| `--conf` | `0.25` | YOLO confidence threshold |
| `--tile-size` | `640` | Tile size in pixels |
| `--overlap` | `0.10` | Overlap fraction between tiles |
| `--min-mask-area` | `50` | Minimum SAM mask area to keep (px) |

## How it works

1. Opens the orthomosaic with rasterio and computes a non-overlapping tile grid
2. For each tile: reads RGB, runs YOLO, filters detections by centre position to avoid double-counting on overlaps
3. Passes kept bounding boxes to SAM as prompts → instance masks
4. Stitches masks into a memory-mapped array covering the full raster extent
5. Writes binary mask GeoTIFF, colour overlay GeoTIFF, and preview PNG

## Output

```
output_dir/
├── orthomosaic_mask.tif            # binary mask (georeferenced)
├── orthomosaic_overlay.tif         # colour overlay (georeferenced)
├── orthomosaic_overlay_preview.png # downsampled preview
└── run_summary.json                # tile stats + total detection count
```

## Previous run results (ggs-orthophoto.tif, 14929×22012 px)

| Metric | Value |
|--------|-------|
| Tiles processed | 1,014 |
| Tiles with detections | 253 (25%) |
| YOLO detections | 15,525 |
| SAM masks stitched | 15,364 (99% yield) |
