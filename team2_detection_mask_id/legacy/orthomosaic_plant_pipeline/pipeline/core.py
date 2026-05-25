from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.windows import Window
from ultralytics import SAM, YOLO

from .clustering import assign_row_col_within_sector, cluster_plants
from .plant_ids import assign_plant_ids
from .viewer import generate_viewer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "inputs"
DEFAULT_WEIGHTS_DIR = ROOT / "weights"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
ORTHO_EXTENSIONS = {".tif", ".tiff"}


@dataclass(frozen=True)
class TileJob:
    tile_id: int
    x0: int
    y0: int
    width: int
    height: int
    padded_width: int
    padded_height: int
    accept_x_min: int
    accept_x_max: int
    accept_y_min: int
    accept_y_max: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full orthomosaic plant pipeline: tile the raster, detect plants with YOLO, "
            "segment with SAM, build plant instances, assign sector-aware IDs, and generate an HTML viewer."
        )
    )
    parser.add_argument("--source", help="Path to the orthomosaic GeoTIFF. If omitted, one TIFF is auto-selected from inputs/.")
    parser.add_argument("--yolo-weights", default=str(DEFAULT_WEIGHTS_DIR / "best.pt"), help="Path to the YOLO weights file.")
    parser.add_argument("--sam-weights", default=str(DEFAULT_WEIGHTS_DIR / "sam_b.pt"), help="Path to the SAM weights file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where the pipeline outputs will be written.")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile width and height in pixels.")
    parser.add_argument("--overlap", type=float, default=0.10, help="Tile overlap fraction.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=max(1, min(4, max((os.cpu_count() or 2) - 1, 1))),
        help="Background worker threads that prefetch and preprocess upcoming tiles while inference runs.",
    )
    parser.add_argument("--min-mask-area", type=int, default=50, help="Discard very small mask fragments.")
    parser.add_argument("--min-instance-area", type=int, default=80, help="Discard very small plant instances.")
    parser.add_argument("--min-cluster-size", type=int, default=25, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--preview-max-dim", type=int, default=4096, help="Maximum preview dimension in pixels.")
    parser.add_argument("--limit-tiles", type=int, default=0, help="Optional debug tile limit. 0 means all tiles.")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep temporary memmap files instead of deleting them.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all stages even if reusable outputs already exist.",
    )
    parser.add_argument(
        "--force-segmentation",
        action="store_true",
        help="Recompute the segmentation stage even if its outputs already exist.",
    )
    return parser.parse_args()


def resolve_source(source_arg: str | None) -> Path:
    if source_arg:
        source = Path(source_arg).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Orthomosaic not found: {source}")
        return source

    candidates = sorted(
        path for path in DEFAULT_INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in ORTHO_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError("No orthomosaic TIFF found. Add a .tif/.tiff file to inputs/ or pass --source.")
    if len(candidates) > 1:
        raise RuntimeError("Multiple orthomosaics found in inputs/. Pass --source explicitly.")
    return candidates[0].resolve()


def validate_inputs(source: Path, yolo_weights: Path, sam_weights: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Orthomosaic not found: {source}")
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
    if not sam_weights.exists():
        raise FileNotFoundError(f"SAM weights not found: {sam_weights}")


def compute_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_tile_jobs(width: int, height: int, tile_size: int, overlap: float) -> list[TileJob]:
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    x_starts = compute_starts(width, tile_size, stride)
    y_starts = compute_starts(height, tile_size, stride)
    jobs: list[TileJob] = []
    tile_id = 0

    for row_index, y0 in enumerate(y_starts):
        y1 = min(y0 + tile_size, height)
        next_y = y_starts[row_index + 1] if row_index + 1 < len(y_starts) else height
        for col_index, x0 in enumerate(x_starts):
            x1 = min(x0 + tile_size, width)
            next_x = x_starts[col_index + 1] if col_index + 1 < len(x_starts) else width
            jobs.append(
                TileJob(
                    tile_id=tile_id,
                    x0=x0,
                    y0=y0,
                    width=x1 - x0,
                    height=y1 - y0,
                    padded_width=tile_size,
                    padded_height=tile_size,
                    accept_x_min=x0,
                    accept_x_max=next_x,
                    accept_y_min=y0,
                    accept_y_max=next_y,
                )
            )
            tile_id += 1
    return jobs


def to_uint8(tile: np.ndarray) -> np.ndarray:
    if tile.dtype == np.uint8:
        return tile
    if np.issubdtype(tile.dtype, np.integer):
        dtype_max = np.iinfo(tile.dtype).max
        if dtype_max <= 255:
            return tile.astype(np.uint8)
        scaled = np.clip(tile.astype(np.float32) / dtype_max * 255.0, 0, 255)
        return scaled.astype(np.uint8)

    finite_tile = np.nan_to_num(tile.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(finite_tile.min())
    max_value = float(finite_tile.max())
    if max_value <= min_value:
        return np.zeros_like(finite_tile, dtype=np.uint8)
    scaled = (finite_tile - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def read_rgb_tile(dataset: rasterio.io.DatasetReader, job: TileJob) -> np.ndarray:
    window = Window(job.x0, job.y0, job.width, job.height)
    band_indexes = [1, 2, 3] if dataset.count >= 3 else [1, 1, 1]
    tile = dataset.read(band_indexes, window=window)
    tile = np.transpose(tile, (1, 2, 0))
    tile = to_uint8(tile)

    if tile.shape[0] == job.padded_height and tile.shape[1] == job.padded_width:
        return tile

    padded = np.zeros((job.padded_height, job.padded_width, 3), dtype=np.uint8)
    padded[: job.height, : job.width] = tile[:, :, :3]
    return padded


def read_rgb_tile_from_source(source_path: Path, job: TileJob) -> np.ndarray:
    with rasterio.open(source_path) as dataset:
        return read_rgb_tile(dataset, job)


def keep_detection(job: TileJob, box_xyxy: np.ndarray) -> bool:
    center_x = float((box_xyxy[0] + box_xyxy[2]) / 2.0) + job.x0
    center_y = float((box_xyxy[1] + box_xyxy[3]) / 2.0) + job.y0
    return (
        job.accept_x_min <= center_x < job.accept_x_max
        and job.accept_y_min <= center_y < job.accept_y_max
    )


def clean_mask(mask: np.ndarray, min_mask_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) < min_mask_area:
        return np.zeros_like(binary, dtype=np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(binary, dtype=np.uint8)

    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) >= min_mask_area:
            cv2.drawContours(cleaned, [contour], contourIdx=-1, color=1, thickness=-1)
    return cleaned


def apply_tile_masks(
    stitched_mask: np.memmap,
    stitched_counts: np.memmap,
    tile_mask_stack: np.ndarray,
    job: TileJob,
    min_mask_area: int,
) -> int:
    accepted = 0
    y_slice = slice(job.y0, job.y0 + job.height)
    x_slice = slice(job.x0, job.x0 + job.width)
    for mask in tile_mask_stack:
        cleaned = clean_mask(mask[: job.height, : job.width], min_mask_area=min_mask_area)
        if cleaned.max() == 0:
            continue
        stitched_mask[y_slice, x_slice] = np.maximum(stitched_mask[y_slice, x_slice], cleaned)
        stitched_counts[y_slice, x_slice] += cleaned.astype(np.uint16)
        accepted += 1
    return accepted


def write_single_band_raster(source_path: Path, destination: Path, array: np.ndarray, dtype: str) -> None:
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype=dtype, nodata=0, compress="LZW")
        with rasterio.open(destination, "w", **profile) as dst:
            dst.write(np.asarray(array, dtype=np.dtype(dtype)), 1)


def blend_overlay(rgb_tile: np.ndarray, mask_tile: np.ndarray) -> np.ndarray:
    overlay = rgb_tile.copy()
    mask_bool = mask_tile > 0
    if not mask_bool.any():
        return overlay
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 255
    overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.40, tint[mask_bool], 0.60, 0)
    return overlay


def write_overlay_raster(source_path: Path, overlay_path: Path, mask_array: np.ndarray) -> None:
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(count=3, dtype="uint8", compress="LZW")
        with rasterio.open(overlay_path, "w", **profile) as dst:
            block_height = 1024
            for y0 in range(0, src.height, block_height):
                height = min(block_height, src.height - y0)
                window = Window(0, y0, src.width, height)
                tile = src.read([1, 2, 3] if src.count >= 3 else [1, 1, 1], window=window)
                tile = np.transpose(tile, (1, 2, 0))
                tile = to_uint8(tile)
                overlay = blend_overlay(tile, np.asarray(mask_array[y0 : y0 + height, :], dtype=np.uint8))
                dst.write(np.transpose(overlay, (2, 0, 1)), window=window)


def write_preview_png(source_path: Path, preview_path: Path, mask_array: np.ndarray, preview_max_dim: int) -> dict:
    with rasterio.open(source_path) as src:
        scale = min(preview_max_dim / src.width, preview_max_dim / src.height, 1.0)
        preview_width = max(1, int(round(src.width * scale)))
        preview_height = max(1, int(round(src.height * scale)))
        tile = src.read(
            [1, 2, 3] if src.count >= 3 else [1, 1, 1],
            out_shape=(3, preview_height, preview_width),
            resampling=rasterio.enums.Resampling.bilinear,
        )
        preview_rgb = np.transpose(tile, (1, 2, 0))
        preview_rgb = to_uint8(preview_rgb)

    mask_uint8 = np.asarray(mask_array, dtype=np.uint8) * 255
    preview_mask = cv2.resize(mask_uint8, (preview_width, preview_height), interpolation=cv2.INTER_NEAREST)
    preview_overlay = blend_overlay(preview_rgb, preview_mask)
    cv2.imwrite(str(preview_path), cv2.cvtColor(preview_overlay, cv2.COLOR_RGB2BGR))
    return {"preview_width": preview_width, "preview_height": preview_height}


def _encode_row_runs(row: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate(row.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append([start, index - 1])
            start = None
    if start is not None:
        runs.append([start, int(row.shape[0]) - 1])
    return runs


def _encode_binary_mask(mask: np.ndarray) -> dict:
    rows: list[dict] = []
    for y_index in range(mask.shape[0]):
        mask_runs = _encode_row_runs(mask[y_index])
        if mask_runs:
            rows.append({"y": y_index, "x_runs": mask_runs})
    return {
        "format": "row_runs_binary_v1",
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "rows": rows,
    }


def _encode_value_runs(row: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    start: int | None = None
    current_value: int | None = None
    for index, value in enumerate(row.tolist()):
        value_int = int(value)
        if value_int == 0:
            if start is not None and current_value is not None:
                runs.append([start, index - 1, current_value])
                start = None
                current_value = None
            continue
        if start is None:
            start = index
            current_value = value_int
            continue
        if value_int != current_value:
            runs.append([start, index - 1, int(current_value)])
            start = index
            current_value = value_int
    if start is not None and current_value is not None:
        runs.append([start, int(row.shape[0]) - 1, int(current_value)])
    return runs


def export_segmentation_json(stage_dir: Path, binary_mask: np.ndarray, count_mask: np.ndarray) -> tuple[Path, Path]:
    binary_payload = _encode_binary_mask(binary_mask)
    count_rows: list[dict] = []

    for y_index in range(count_mask.shape[0]):
        count_runs = _encode_value_runs(count_mask[y_index])
        if count_runs:
            count_rows.append({"y": y_index, "x_runs": count_runs})

    count_payload = {
        "format": "row_runs_counts_v1",
        "width": int(count_mask.shape[1]),
        "height": int(count_mask.shape[0]),
        "rows": count_rows,
    }

    binary_json_path = stage_dir / "orthomosaic_binary_mask.json"
    overlap_json_path = stage_dir / "orthomosaic_overlap_count.json"
    write_json(binary_json_path, binary_payload)
    write_json(overlap_json_path, count_payload)
    return binary_json_path, overlap_json_path


def ensure_segmentation_json_exports(stage_dir: Path, binary_mask_path: Path, overlap_count_path: Path) -> tuple[Path, Path]:
    binary_json_path = stage_dir / "orthomosaic_binary_mask.json"
    overlap_json_path = stage_dir / "orthomosaic_overlap_count.json"
    if binary_json_path.exists() and overlap_json_path.exists():
        return binary_json_path, overlap_json_path

    with rasterio.open(binary_mask_path) as mask_src:
        binary_mask = mask_src.read(1).astype(np.uint8)
    with rasterio.open(overlap_count_path) as count_src:
        count_mask = count_src.read(1).astype(np.uint16)
    return export_segmentation_json(stage_dir, binary_mask, count_mask)


def run_segmentation(args: SimpleNamespace, stage_dir: Path) -> dict:
    source = Path(args.source).resolve()
    yolo_weights = Path(args.yolo_weights).resolve()
    sam_weights = Path(args.sam_weights).resolve()
    validate_inputs(source, yolo_weights, sam_weights)
    stage_dir.mkdir(parents=True, exist_ok=True)

    yolo_model = YOLO(str(yolo_weights))
    sam_model = SAM(str(sam_weights))
    prefetch_workers = max(0, int(getattr(args, "prefetch_workers", 0)))

    with rasterio.open(source) as src:
        jobs = build_tile_jobs(src.width, src.height, tile_size=args.tile_size, overlap=args.overlap)
        if args.limit_tiles > 0:
            jobs = jobs[: args.limit_tiles]

        mask_memmap_path = stage_dir / "stitched_mask.dat"
        count_memmap_path = stage_dir / "stitched_mask_counts.dat"
        stitched_mask = np.memmap(mask_memmap_path, mode="w+", dtype=np.uint8, shape=(src.height, src.width))
        stitched_counts = np.memmap(count_memmap_path, mode="w+", dtype=np.uint16, shape=(src.height, src.width))
        stitched_mask[:] = 0
        stitched_counts[:] = 0

        total_detections = 0
        total_segmented = 0
        tile_summaries: list[dict] = []
        future_by_job_id: dict[int, Future[np.ndarray]] = {}
        executor: ThreadPoolExecutor | None = None

        try:
            if prefetch_workers > 0:
                executor = ThreadPoolExecutor(max_workers=prefetch_workers)
                initial_prefetch = min(len(jobs), prefetch_workers + 1)
                for job in jobs[:initial_prefetch]:
                    future_by_job_id[job.tile_id] = executor.submit(read_rgb_tile_from_source, source, job)

            for index, job in enumerate(jobs, start=1):
                if executor is not None:
                    rgb_tile = future_by_job_id.pop(job.tile_id).result()
                    next_prefetch_index = index - 1 + prefetch_workers + 1
                    if next_prefetch_index < len(jobs):
                        next_job = jobs[next_prefetch_index]
                        future_by_job_id[next_job.tile_id] = executor.submit(read_rgb_tile_from_source, source, next_job)
                else:
                    rgb_tile = read_rgb_tile(src, job)

                detections = yolo_model.predict(source=rgb_tile, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
                boxes = detections.boxes
                if boxes is None or len(boxes) == 0:
                    tile_summaries.append({"tile_id": job.tile_id, "x0": job.x0, "y0": job.y0, "detections": 0, "segmented_masks": 0})
                    print(f"[{index}/{len(jobs)}] tile {job.tile_id}: no detections")
                    continue

                raw_boxes = boxes.xyxy.cpu().numpy()
                kept_indexes = [i for i, box in enumerate(raw_boxes) if keep_detection(job, box)]
                total_detections += len(kept_indexes)
                if not kept_indexes:
                    tile_summaries.append({"tile_id": job.tile_id, "x0": job.x0, "y0": job.y0, "detections": 0, "segmented_masks": 0})
                    print(f"[{index}/{len(jobs)}] tile {job.tile_id}: overlap-only detections skipped")
                    continue

                prompt_boxes = raw_boxes[kept_indexes].tolist()
                sam_result = sam_model.predict(source=rgb_tile, bboxes=prompt_boxes, device=args.device, verbose=False)[0]
                segmented_masks = 0
                if sam_result.masks is not None:
                    mask_stack = sam_result.masks.data.cpu().numpy()
                    segmented_masks = apply_tile_masks(stitched_mask, stitched_counts, mask_stack, job, args.min_mask_area)
                    total_segmented += segmented_masks

                tile_summaries.append(
                    {"tile_id": job.tile_id, "x0": job.x0, "y0": job.y0, "detections": len(kept_indexes), "segmented_masks": segmented_masks}
                )
                print(f"[{index}/{len(jobs)}] tile {job.tile_id}: {len(kept_indexes)} detections, {segmented_masks} stitched masks")
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        stitched_mask.flush()
        stitched_counts.flush()
        binary_mask = np.array(stitched_mask, dtype=np.uint8, copy=True)
        count_mask = np.array(stitched_counts, dtype=np.uint16, copy=True)

    mask_tif = stage_dir / "orthomosaic_binary_mask.tif"
    overlap_tif = stage_dir / "orthomosaic_overlap_count.tif"
    overlay_tif = stage_dir / "orthomosaic_masked_overlay.tif"
    preview_png = stage_dir / "orthomosaic_masked_overlay_preview.png"
    binary_json, overlap_json = export_segmentation_json(stage_dir, binary_mask, count_mask)
    write_single_band_raster(source, mask_tif, binary_mask, "uint8")
    write_single_band_raster(source, overlap_tif, count_mask, "uint16")
    write_overlay_raster(source, overlay_tif, binary_mask)
    preview_meta = write_preview_png(source, preview_png, binary_mask, preview_max_dim=args.preview_max_dim)

    metadata = {
        "source": str(source),
        "yolo_weights": str(yolo_weights),
        "sam_weights": str(sam_weights),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "num_tiles": len(jobs),
        "total_detections": total_detections,
        "total_segmented_masks": total_segmented,
        "binary_mask_raster": str(mask_tif),
        "binary_mask_json": str(binary_json),
        "overlap_count_raster": str(overlap_tif),
        "overlap_count_json": str(overlap_json),
        "masked_overlay_raster": str(overlay_tif),
        "preview_png": str(preview_png),
        "intermediate_mask_memmap": str(mask_memmap_path),
        "intermediate_count_memmap": str(count_memmap_path),
        "prefetch_workers": prefetch_workers,
        "tile_summary": tile_summaries,
        **preview_meta,
    }

    if not args.keep_intermediates:
        del binary_mask
        del count_mask
        del stitched_mask
        del stitched_counts
        cleanup_warnings: list[str] = []
        for temp_path in (mask_memmap_path, count_memmap_path):
            try:
                temp_path.unlink(missing_ok=True)
            except PermissionError:
                cleanup_warnings.append(
                    f"Could not delete temporary file because it is still locked by Windows: {temp_path}"
                )
            except OSError as exc:
                cleanup_warnings.append(f"Could not delete temporary file {temp_path}: {exc}")

        if not cleanup_warnings:
            metadata["intermediate_mask_memmap"] = None
            metadata["intermediate_count_memmap"] = None
        else:
            metadata["cleanup_warnings"] = cleanup_warnings

    return metadata


def extract_instances(
    source_path: Path,
    binary_mask_path: Path,
    output_dir: Path,
    min_instance_area: int,
) -> tuple[list[dict], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(binary_mask_path) as mask_src:
        binary_mask = mask_src.read(1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    labels = labels.astype(np.uint32)

    with rasterio.open(source_path) as src:
        transform = src.transform
        source_crs = src.crs.to_string() if src.crs else None
        source_width = src.width
        source_height = src.height

    instance_records: list[dict] = []
    filtered_labels = np.zeros_like(labels, dtype=np.uint32)
    next_label = 1

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_instance_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        centroid_x = float(centroids[label][0])
        centroid_y = float(centroids[label][1])
        geo_x, geo_y = xy(transform, centroid_y, centroid_x, offset="center")

        component_mask = (labels[y : y + height, x : x + width] == label).astype(np.uint8) * 255
        filtered_labels[labels == label] = next_label
        instance_records.append(
            {
                "instance_id": next_label,
                "pixel_x": int(round(centroid_x)),
                "pixel_y": int(round(centroid_y)),
                "geo_x": float(geo_x),
                "geo_y": float(geo_y),
                "area_px": area,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": width,
                "bbox_height": height,
                "mask_array": component_mask,
                "source_crs": source_crs,
                "source_width": source_width,
                "source_height": source_height,
            }
        )
        next_label += 1

    instance_raster = output_dir / "plant_instance_labels.tif"
    write_single_band_raster(source_path, instance_raster, filtered_labels, "uint32")
    return instance_records, instance_raster


def write_per_plant_masks(plants: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_payload: list[dict] = []
    for plant in plants:
        plant_id = plant["plant_id"]
        mask_path = output_dir / f"{plant_id}_mask.png"
        cv2.imwrite(str(mask_path), plant["mask_array"])
        plant["mask_file"] = mask_path.name
        index_payload.append(
            {
                "plant_id": plant_id,
                "instance_id": plant["instance_id"],
                "mask_file": mask_path.name,
                "bbox": {
                    "x": plant["bbox_x"],
                    "y": plant["bbox_y"],
                    "width": plant["bbox_width"],
                    "height": plant["bbox_height"],
                },
            }
        )

    index_path = output_dir / "mask_index.json"
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    return index_path


def _round_value(value):
    if isinstance(value, float):
        return round(value, 6)
    return value


def _public_plant_record(plant: dict) -> dict:
    return {key: _round_value(value) for key, value in plant.items() if key not in {"mask_array"}}


def _instance_stage_record(plant: dict) -> dict:
    public_record = _public_plant_record(plant)
    public_record["mask"] = _encode_binary_mask((plant["mask_array"] > 0).astype(np.uint8))
    return public_record


def _pre_id_instance_record(instance: dict) -> dict:
    return {
        "instance_id": int(instance["instance_id"]),
        "pixel_x": int(instance["pixel_x"]),
        "pixel_y": int(instance["pixel_y"]),
        "geo_x": _round_value(float(instance["geo_x"])),
        "geo_y": _round_value(float(instance["geo_y"])),
        "area_px": int(instance["area_px"]),
        "bbox_x": int(instance["bbox_x"]),
        "bbox_y": int(instance["bbox_y"]),
        "bbox_width": int(instance["bbox_width"]),
        "bbox_height": int(instance["bbox_height"]),
        "source_crs": instance.get("source_crs"),
        "source_width": int(instance["source_width"]),
        "source_height": int(instance["source_height"]),
        "mask": _encode_binary_mask((instance["mask_array"] > 0).astype(np.uint8)),
    }


def write_pre_id_instance_json(instances: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "plant_instances_pre_id.json"
    write_json(path, [_pre_id_instance_record(instance) for instance in instances])
    return path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, plants: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "plant_id",
        "instance_id",
        "sector_id",
        "sector_label",
        "row_index",
        "col_index",
        "pixel_x",
        "pixel_y",
        "geo_x",
        "geo_y",
        "area_px",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "mask_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for plant in plants:
            writer.writerow({name: plant.get(name) for name in fieldnames})


def write_coordinates_csv(path: Path, plants: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["plant_id", "instance_id", "pixel_x", "pixel_y", "geo_x", "geo_y"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for plant in plants:
            writer.writerow({name: plant.get(name) for name in fieldnames})


def write_ids_csv(path: Path, plants: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["plant_id", "instance_id", "sector_id", "sector_label", "row_index", "col_index", "geo_hash"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for plant in plants:
            writer.writerow({name: plant.get(name) for name in fieldnames})


def write_geojson(path: Path, plants: list[dict]) -> None:
    features = []
    for plant in plants:
        properties = {
            "plant_id": plant["plant_id"],
            "instance_id": plant["instance_id"],
            "sector_id": plant["sector_id"],
            "sector_label": plant["sector_label"],
            "row_index": plant["row_index"],
            "col_index": plant["col_index"],
            "pixel_x": plant["pixel_x"],
            "pixel_y": plant["pixel_y"],
            "area_px": plant["area_px"],
            "mask_file": plant["mask_file"],
        }
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(float(plant["geo_x"]), 6), round(float(plant["geo_y"]), 6)],
                },
            }
        )

    payload = {"type": "FeatureCollection", "features": features}
    write_json(path, payload)


def _segmentation_outputs_exist(stage_dir: Path) -> bool:
    required = [
        stage_dir / "orthomosaic_binary_mask.tif",
        stage_dir / "orthomosaic_overlap_count.tif",
        stage_dir / "orthomosaic_masked_overlay.tif",
        stage_dir / "orthomosaic_masked_overlay_preview.png",
    ]
    return all(path.exists() for path in required)


def load_segmentation_metadata(
    source_path: Path,
    yolo_weights: Path,
    sam_weights: Path,
    stage_dir: Path,
    tile_size: int,
    overlap: float,
) -> dict:
    preview_path = stage_dir / "orthomosaic_masked_overlay_preview.png"
    binary_mask_path = stage_dir / "orthomosaic_binary_mask.tif"
    overlap_count_path = stage_dir / "orthomosaic_overlap_count.tif"
    binary_json_path, overlap_json_path = ensure_segmentation_json_exports(
        stage_dir=stage_dir,
        binary_mask_path=binary_mask_path,
        overlap_count_path=overlap_count_path,
    )
    preview = cv2.imread(str(preview_path))
    if preview is None:
        raise RuntimeError(f"Failed to read preview image from existing segmentation output: {preview_path}")
    preview_height, preview_width = preview.shape[:2]

    return {
        "source": str(source_path),
        "yolo_weights": str(yolo_weights),
        "sam_weights": str(sam_weights),
        "tile_size": tile_size,
        "overlap": overlap,
        "binary_mask_raster": str(binary_mask_path),
        "binary_mask_json": str(binary_json_path),
        "overlap_count_raster": str(overlap_count_path),
        "overlap_count_json": str(overlap_json_path),
        "masked_overlay_raster": str(stage_dir / "orthomosaic_masked_overlay.tif"),
        "preview_png": str(preview_path),
        "preview_width": preview_width,
        "preview_height": preview_height,
        "intermediate_mask_memmap": str(stage_dir / "stitched_mask.dat") if (stage_dir / "stitched_mask.dat").exists() else None,
        "intermediate_count_memmap": str(stage_dir / "stitched_mask_counts.dat") if (stage_dir / "stitched_mask_counts.dat").exists() else None,
        "reused_existing_outputs": True,
    }


def _final_outputs_exist(output_dir: Path) -> bool:
    required = [
        output_dir / "run_summary.json",
        output_dir / "02_instances" / "plant_instance_labels.tif",
        output_dir / "03_ids" / "plants_with_ids.csv",
        output_dir / "03_ids" / "plants_with_ids.json",
        output_dir / "03_ids" / "plants_points.geojson",
        output_dir / "03_ids" / "plant_coordinates.csv",
        output_dir / "03_ids" / "plant_ids.csv",
        output_dir / "04_masks" / "mask_index.json",
        output_dir / "05_viewer" / "viewer.html",
    ]
    return all(path.exists() for path in required)


def enrich_for_preview(plants: list[dict], preview_width: int, preview_height: int, source_width: int, source_height: int) -> list[dict]:
    scale_x = preview_width / max(source_width, 1)
    scale_y = preview_height / max(source_height, 1)
    enriched: list[dict] = []
    for plant in plants:
        updated = dict(plant)
        updated["preview_x"] = round(float(plant["pixel_x"]) * scale_x, 2)
        updated["preview_y"] = round(float(plant["pixel_y"]) * scale_y, 2)
        enriched.append(updated)
    return enriched


def export_outputs(
    source_path: Path,
    segmentation_meta: dict,
    plants: list[dict],
    output_dir: Path,
    instance_raster: Path,
) -> dict:
    stage_instances = output_dir / "02_instances"
    stage_ids = output_dir / "03_ids"
    stage_masks = output_dir / "04_masks"
    stage_viewer = output_dir / "05_viewer"
    stage_instances.mkdir(parents=True, exist_ok=True)
    stage_ids.mkdir(parents=True, exist_ok=True)
    stage_viewer.mkdir(parents=True, exist_ok=True)

    instance_json = stage_instances / "plant_instances.json"
    instance_csv = stage_instances / "plant_instances.csv"
    ids_json = stage_ids / "plants_with_ids.json"
    ids_csv = stage_ids / "plants_with_ids.csv"
    coordinates_csv = stage_ids / "plant_coordinates.csv"
    ids_only_csv = stage_ids / "plant_ids.csv"
    points_geojson = stage_ids / "plants_points.geojson"
    mask_index = write_per_plant_masks(plants, stage_masks)

    instance_records = [_instance_stage_record(plant) for plant in plants]
    public_records = [_public_plant_record(plant) for plant in plants]
    write_json(instance_json, instance_records)
    write_csv(instance_csv, public_records)
    write_json(ids_json, public_records)
    write_csv(ids_csv, public_records)
    write_coordinates_csv(coordinates_csv, public_records)
    write_ids_csv(ids_only_csv, public_records)
    write_geojson(points_geojson, public_records)

    preview_png_path = Path(segmentation_meta["preview_png"])
    viewer_preview_path = stage_viewer / preview_png_path.name
    if preview_png_path.resolve() != viewer_preview_path.resolve():
        shutil.copyfile(preview_png_path, viewer_preview_path)

    viewer_path = generate_viewer(
        plants=public_records,
        metadata={
            "pipeline_name": "orthomosaic_plant_pipeline",
            "source": str(source_path),
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "preview_width": segmentation_meta["preview_width"],
            "preview_height": segmentation_meta["preview_height"],
        },
        preview_image_name=viewer_preview_path.name,
        output_path=stage_viewer / "viewer.html",
    )

    plants_by_sector = Counter(plant["sector_label"] for plant in public_records if int(plant["sector_id"]) != -1)
    summary = {
        "pipeline_name": "orthomosaic_plant_pipeline",
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": str(source_path),
        "binary_mask_raster": segmentation_meta["binary_mask_raster"],
        "masked_overlay_raster": segmentation_meta["masked_overlay_raster"],
        "instance_raster": str(instance_raster),
        "preview_png": str(preview_png_path),
        "viewer_html": str(viewer_path),
        "mask_index_json": str(mask_index),
        "plants_csv": str(ids_csv),
        "coordinates_csv": str(coordinates_csv),
        "ids_csv": str(ids_only_csv),
        "plants_json": str(ids_json),
        "plants_geojson": str(points_geojson),
        "total_plants": len(public_records),
        "noise_plants": sum(1 for plant in public_records if int(plant["sector_id"]) == -1),
        "plants_by_sector": dict(sorted(plants_by_sector.items())),
        "segmentation": segmentation_meta,
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary


def run_pipeline(args: argparse.Namespace) -> dict:
    source_path = resolve_source(args.source)
    yolo_weights = Path(args.yolo_weights).resolve()
    sam_weights = Path(args.sam_weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir = output_dir / "01_segmentation"

    if not args.force and _final_outputs_exist(output_dir):
        summary = load_json(output_dir / "run_summary.json")
        summary["reused_existing_outputs"] = True
        return summary

    reuse_segmentation = not args.force and not args.force_segmentation and _segmentation_outputs_exist(segmentation_dir)
    if reuse_segmentation:
        print(f"Reusing existing segmentation outputs from: {segmentation_dir}")
        segmentation_meta = load_segmentation_metadata(
            source_path=source_path,
            yolo_weights=yolo_weights,
            sam_weights=sam_weights,
            stage_dir=segmentation_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
    else:
        segmentation_args = SimpleNamespace(
            source=str(source_path),
            yolo_weights=str(yolo_weights),
            sam_weights=str(sam_weights),
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            prefetch_workers=args.prefetch_workers,
            min_mask_area=args.min_mask_area,
            preview_max_dim=args.preview_max_dim,
            limit_tiles=args.limit_tiles,
            keep_intermediates=args.keep_intermediates,
        )
        segmentation_meta = run_segmentation(segmentation_args, segmentation_dir)

    instances, instance_raster = extract_instances(
        source_path=source_path,
        binary_mask_path=Path(segmentation_meta["binary_mask_raster"]),
        output_dir=output_dir / "02_instances",
        min_instance_area=args.min_instance_area,
    )
    write_pre_id_instance_json(instances, output_dir / "02_instances")

    if not instances:
        summary = {
            "pipeline_name": "orthomosaic_plant_pipeline",
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": str(source_path),
            "total_plants": 0,
            "viewer_html": None,
            "segmentation": segmentation_meta,
        }
        write_json(output_dir / "run_summary.json", summary)
        return summary

    plants = cluster_plants(instances, min_cluster_size=args.min_cluster_size)
    plants = assign_row_col_within_sector(plants)
    plants = assign_plant_ids(plants)
    plants = enrich_for_preview(
        plants,
        preview_width=segmentation_meta["preview_width"],
        preview_height=segmentation_meta["preview_height"],
        source_width=instances[0]["source_width"],
        source_height=instances[0]["source_height"],
    )
    return export_outputs(source_path, segmentation_meta, plants, output_dir, instance_raster)


def main() -> None:
    args = parse_args()
    summary = run_pipeline(args)
    print("Pipeline complete.")
    print(f"Total plants: {summary.get('total_plants', 0)}")
    if summary.get("viewer_html"):
        print(f"Viewer: {summary['viewer_html']}")
    print(f"Run summary: {Path(args.output_dir).resolve() / 'run_summary.json'}")
