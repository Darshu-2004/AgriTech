# Latest Masking Pipeline Documentation

## Overview

The `latest_masking_pipeline` project is an end-to-end orthomosaic segmentation workflow for pineapple plant mapping.

It takes a large drone orthomosaic GeoTIFF, divides it into overlapping tiles, detects pineapple plants with a custom YOLOv8x model, refines each detection into a segmentation mask with SAM, and stitches all masks back into the full orthomosaic coordinate space.

The final result is a georeferenced mask raster, a georeferenced orthomosaic overlay, and a PNG preview for quick inspection.

## Main objective

The pipeline is designed to solve this workflow in one pass:

1. Read a large orthomosaic without loading the whole file into RAM.
2. Tile the image into `640 x 640` windows with `10%` overlap.
3. Run the trained YOLOv8x detector on each tile.
4. Use YOLO bounding boxes as prompts for SAM segmentation.
5. Paste each tile mask back into the exact global location in the original orthomosaic.
6. Export stitched outputs for GIS and visual review.

## Project files

- `run_full_orthomosaic_pipeline.py`
  This is the one-command entrypoint.

- `orthomosaic_mask_pipeline.py`
  This contains the main tiling, inference, stitching, and export logic.

- `run_pipeline.cmd`
  A Windows launcher that runs the entrypoint script.

- `requirements.txt`
  Python dependencies needed by the pipeline.

- `weights/best.pt`
  Local YOLOv8x trained detector weights.

- `weights/sam_b.pt`
  Local SAM model weights.

- `ortho/ggs-orthophoto.tif`
  Input orthomosaic source image.

## One-command execution

From the project root:

```powershell
python .\run_full_orthomosaic_pipeline.py
```

Or on Windows:

```text
run_pipeline.cmd
```

## High-level architecture

The pipeline has four major stages:

1. Input discovery and configuration
2. Tile generation
3. Tile-wise detection and segmentation
4. Stitched output generation

## Stage 1: Input discovery and configuration

This stage is handled mainly by `run_full_orthomosaic_pipeline.py`.

### What it does

- Finds the orthomosaic automatically from the `ortho/` folder if `--source` is not provided.
- Uses local defaults for:
  - YOLO weights: `weights/best.pt`
  - SAM weights: `weights/sam_b.pt`
  - output folder: `pipeline_outputs/orthomosaic_full`
- Builds a simple argument bundle and passes it into `run_pipeline()`.

### Why this wrapper exists

The wrapper keeps day-to-day usage simple. The heavy implementation stays in `orthomosaic_mask_pipeline.py`, while the top-level script provides a single clean command for normal use.

## Stage 2: Orthomosaic tiling

This stage is implemented in `orthomosaic_mask_pipeline.py`.

### Why tiling is necessary

The orthomosaic is too large to process efficiently as one image. Tiling is necessary because:

- YOLO and SAM are designed for manageable image sizes.
- GPU memory is limited.
- windowed reading is much more memory-efficient for large GeoTIFFs.

### Tile size and overlap

Default configuration:

- Tile size: `640 x 640`
- Overlap: `10%`

This means the stride is:

```text
stride = tile_size * (1 - overlap)
```

For the default settings:

```text
stride = 640 * 0.90 = 576
```

### Relevant functions

- `compute_starts(length, tile_size, stride)`
  Computes the start positions for tiles along one axis.

- `build_tile_jobs(width, height, tile_size, overlap)`
  Builds the full list of tile windows.

### TileJob structure

Each tile is represented by the `TileJob` dataclass:

- `tile_id`
- `x0`, `y0`
- `width`, `height`
- `padded_width`, `padded_height`
- `accept_x_min`, `accept_x_max`
- `accept_y_min`, `accept_y_max`

The most important values are:

- `x0`, `y0`
  The top-left position of the tile in the full orthomosaic.

- `width`, `height`
  The real size of the tile read from the source image.

- `accept_*`
  The valid ownership area for detections, used to avoid duplicates from tile overlap.

## Stage 3: Reading tiles from the orthomosaic

### Raster access strategy

The pipeline uses `rasterio` and `Window()` objects to read only the needed portion of the GeoTIFF.

This is important because:

- it avoids loading the full orthomosaic into memory
- it preserves exact pixel placement
- it works well with georeferenced rasters

### Relevant functions

- `read_rgb_tile(dataset, job)`
  Reads the tile window from the orthomosaic.

- `to_uint8(tile)`
  Converts imagery from the source raster datatype into `uint8` for model inference and visualization.

### Handling datatype differences

Many orthomosaics are not stored as `uint8`. They may be `uint16` or floating point.

The function `to_uint8()` normalizes the source data into a range suitable for YOLO and SAM inference.

### Edge tile padding

Tiles at the image boundary may be smaller than `640 x 640`.

To keep model input sizes consistent, these edge tiles are padded with zeros to the configured tile size.

The original tile width and height are still preserved so only valid pixels are stitched back.

## Stage 4: Detection with YOLOv8x

### Model loading

The YOLO detector is loaded once:

```python
yolo_model = YOLO(str(yolo_weights))
```

This avoids repeated model initialization for every tile.

### Per-tile inference

For each tile:

```python
detections = yolo_model.predict(
    source=rgb_tile,
    conf=args.conf,
    imgsz=args.imgsz,
    device=args.device,
    verbose=False,
)[0]
```

### Outputs used from YOLO

The pipeline uses:

- `detections.boxes.xyxy`

These are bounding boxes in tile-local pixel coordinates:

```text
[x1, y1, x2, y2]
```

### Confidence threshold

Default confidence threshold:

- `0.25`

This is configurable through `--conf`.

## Stage 5: Duplicate control across overlapping tiles

### Problem

Because tiles overlap, the same plant can appear in more than one tile.

Without duplicate handling, the stitched output would contain repeated masks for the same object.

### Current solution

The pipeline solves this with tile ownership by detection center.

### How it works

For each YOLO box:

1. The box center is computed.
2. The local center is converted into global orthomosaic coordinates.
3. The detection is kept only if the center falls inside the tile's ownership region.

This logic is implemented in:

- `keep_detection(job, box_xyxy)`

### Why this helps

This method gives each object one preferred tile even though the object may appear in multiple overlapping tiles.

That significantly reduces duplicate stitched masks while keeping the benefits of overlap near tile boundaries.

## Stage 6: Segmentation with SAM

### Prompting strategy

YOLO detections are not used as final outputs directly. Instead, the detector provides box prompts for SAM.

For each tile, only the kept detection boxes are sent into SAM:

```python
sam_result = sam_model.predict(
    source=rgb_tile,
    bboxes=prompt_boxes,
    device=args.device,
    verbose=False,
)[0]
```

### Why this two-stage design works well

- YOLO is fast and good at locating objects.
- SAM is good at producing detailed object masks from prompts.

This combination gives a better plant outline than using only detection boxes.

## Stage 7: Mask cleanup

Raw SAM masks are cleaned before stitching.

### Cleanup function

- `clean_mask(mask, min_mask_area)`

### What it does

1. Converts the SAM output into a binary mask.
2. Removes tiny mask fragments.
3. Finds external contours.
4. Fills only contours that exceed the area threshold.

### Minimum mask area

Default:

- `50` pixels

This helps reject tiny noise fragments and unstable mask speckles.

## Stage 8: Stitching masks back to the orthomosaic

### Core idea

Every accepted SAM mask is written back to the full orthomosaic at the exact location from which its tile was read.

### Relevant function

- `apply_tile_masks(stitched_mask, stitched_counts, tile_mask_stack, job, min_mask_area)`

### How the stitch works

For each accepted mask:

1. Crop the mask to the original tile height and width.
2. Clean the mask.
3. Compute the global slices:
   - `y_slice = job.y0 : job.y0 + job.height`
   - `x_slice = job.x0 : job.x0 + job.width`
4. Merge the mask into the stitched raster.

### Stitch buffer design

Two memory-mapped arrays are created:

- `stitched_mask`
  Final binary mask accumulator.

- `stitched_counts`
  Per-pixel overlap count buffer.

### Why memmap is used

The stitched raster is large, so `numpy.memmap` is used to avoid holding the full stitched image purely in RAM.

This makes the workflow practical for large orthomosaics.

### Merge rule

The current merge logic uses:

```python
stitched_mask[y_slice, x_slice] = np.maximum(stitched_mask[y_slice, x_slice], cleaned)
```

This means:

- any positive mask pixel is kept
- overlapping masks are unioned together

The count buffer is also incremented:

```python
stitched_counts[y_slice, x_slice] += cleaned.astype(np.uint16)
```

This can be useful later for overlap diagnostics, although the current exported outputs do not directly visualize it.

## Stage 9: Output generation

After all tiles are processed, the pipeline exports three main outputs.

### 1. Georeferenced binary mask raster

Written by:

- `write_mask_raster(source_path, mask_path, mask_array)`

Output:

- `orthomosaic_mask.tif`

Characteristics:

- single band
- `uint8`
- georeferenced
- compressed with LZW

### 2. Georeferenced orthomosaic overlay raster

Written by:

- `write_overlay_raster(source_path, overlay_path, mask_array)`

Output:

- `orthomosaic_overlay.tif`

Characteristics:

- three bands
- `uint8`
- georeferenced
- original orthomosaic appearance with green mask overlay

### 3. PNG preview

Written by:

- `write_preview_png(source_path, preview_path, mask_array, preview_max_dim)`

Output:

- `orthomosaic_overlay_preview.png`

This is meant for quick viewing outside GIS software.

## Overlay color logic

The mask overlay is generated by `blend_overlay()`.

### Visual style

- mask color: green
- no labels
- no confidence text
- no bounding boxes

This keeps the final result visually clean and focused on segmentation only.

## Run metadata

The pipeline also writes:

- `run_summary.json`

This file contains:

- source file path
- model paths
- tile size
- overlap
- number of tiles
- total detections
- total stitched masks
- output file paths
- per-tile summary

This is useful for auditing and debugging a run.

## CLI parameters

The main implementation in `orthomosaic_mask_pipeline.py` supports these key options:

- `--source`
- `--yolo-weights`
- `--sam-weights`
- `--output-dir`
- `--tile-size`
- `--overlap`
- `--conf`
- `--imgsz`
- `--device`
- `--min-mask-area`
- `--preview-max-dim`
- `--limit-tiles`
- `--keep-intermediates`

The one-command wrapper exposes the most useful subset for normal use.

## Default runtime flow

When you run:

```powershell
python .\run_full_orthomosaic_pipeline.py
```

the actual flow is:

1. discover orthomosaic from `ortho/`
2. load YOLO from `weights/best.pt`
3. load SAM from `weights/sam_b.pt`
4. build tile grid
5. read each tile by raster window
6. run YOLO detection
7. reject overlap-duplicate detections by box-center ownership
8. run SAM on retained boxes
9. clean masks
10. stitch masks into the full orthomosaic mask
11. export GeoTIFF and PNG outputs
12. write run summary
13. optionally remove intermediate memmap files

## Accuracy considerations

The pipeline tries to preserve spatial accuracy in several ways.

### 1. Window-based reads

Tiles are read directly from the orthomosaic by exact raster windows.

### 2. Global coordinate stitching

Masks are written back using each tile's true `x0`, `y0`, `width`, and `height`.

### 3. Edge-aware handling

Edge tiles are padded for inference, but only the valid image area is stitched back.

### 4. Overlap-aware duplicate suppression

Object ownership by box center helps prevent the same plant from being stitched multiple times from neighboring tiles.

## Current limitations

There are a few practical limitations to be aware of.

### Large model and raster files are local assets

The repo excludes:

- `weights/`
- `ortho/`

So the code is versioned, but the heavy models and imagery stay local unless you manage them separately.

### Duplicate reduction is heuristic

The box-center ownership rule is effective, but it is still a heuristic. Very large objects or unstable detections near boundaries can still create small inconsistencies.

### Output overlay is visual, not vector

The stitched output is currently raster-based. This orthomosaic pipeline does not yet export polygons or GeoJSON from the stitched mask.

### Band usage

The current implementation uses the first three raster bands as RGB input. If the orthomosaic has more bands, the extra bands are not used during inference.

## Suggested future improvements

Potential upgrades include:

- export polygons or GeoJSON from the stitched mask
- add non-maximum suppression or mask-level merging across neighboring tiles
- support batch processing of multiple orthomosaics
- support configurable overlay colors
- add optional TIFF pyramids for faster GIS viewing
- add logging to file instead of only console progress
- add unit tests for tiling and stitching utilities

## Dependency summary

Main libraries used:

- `ultralytics`
  Provides YOLO and SAM model interfaces.

- `rasterio`
  Handles georeferenced raster I/O and tiled reads/writes.

- `numpy`
  Used for arrays, memmaps, and raster accumulation.

- `opencv-python`
  Used for mask cleanup, contour operations, and image blending.

- `torch`
  Backend used by Ultralytics models.

## Practical run example

Standard run:

```powershell
python .\run_full_orthomosaic_pipeline.py
```

Example with custom options:

```powershell
python .\run_full_orthomosaic_pipeline.py --device 0 --conf 0.30 --tile-size 640 --overlap 0.10
```

Debug run on a limited number of tiles:

```powershell
python .\orthomosaic_mask_pipeline.py --limit-tiles 10
```

## Output files after a successful run

Typical output folder:

`pipeline_outputs/orthomosaic_full`

Expected files:

- `orthomosaic_mask.tif`
- `orthomosaic_overlay.tif`
- `orthomosaic_overlay_preview.png`
- `run_summary.json`

If `--keep-intermediates` is used, the following can also remain:

- `stitched_mask.dat`
- `stitched_mask_counts.dat`

## Summary

This pipeline is a practical large-image segmentation system built around a strong division of responsibilities:

- `YOLO` finds where plants are.
- `SAM` refines those locations into masks.
- `rasterio` preserves spatial placement across the full orthomosaic.
- `numpy.memmap` keeps large stitched rasters manageable.

The result is a reproducible workflow that turns a large orthomosaic into a stitched, georeferenced plant-mask product with a simple one-command interface.
