# Latest Masking Pipeline

End-to-end pipeline for pineapple plant detection and segmentation on large drone orthomosaics.

## What it does

1. Splits a large orthomosaic into `640x640` tiles with `10%` overlap.
2. Runs a custom YOLOv8x detector on each tile.
3. Uses the YOLO boxes as prompts for SAM to generate segmentation masks.
4. Stitches all tile masks back into the full orthomosaic at the original pixel coordinates.
5. Writes a georeferenced mask raster, a georeferenced overlay raster, and a PNG preview.

## Repo layout

- `run_full_orthomosaic_pipeline.py`: one-command entrypoint
- `orthomosaic_mask_pipeline.py`: tiling, inference, and stitching logic
- `run_pipeline.cmd`: Windows launcher
- `requirements.txt`: Python dependencies
- `weights/`: place local model files here
- `ortho/`: place the input orthomosaic here

## Required local files

These files are intentionally ignored by git because they are large local assets:

- `weights/best.pt`
- `weights/sam_b.pt`
- `ortho/ggs-orthophoto.tif`

## One-command run

From this folder:

```powershell
python .\run_full_orthomosaic_pipeline.py
```

Or on Windows:

```text
run_pipeline.cmd
```

## Output

The pipeline writes outputs under:

`pipeline_outputs/orthomosaic_full`

Main outputs:

- `orthomosaic_overlay.tif`
- `orthomosaic_mask.tif`
- `orthomosaic_overlay_preview.png`
- `run_summary.json`

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
