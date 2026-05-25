from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from ultralytics import SAM, YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_YOLO_WEIGHTS = ROOT / "weights" / "best.pt"
DEFAULT_ORTHO = ROOT / "ortho" / "ggs-orthophoto.tif"


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
            "Tile an orthomosaic, run YOLOv8 + SAM prompted segmentation on each tile, "
            "and stitch the masks back into a georeferenced overlay."
        )
    )
    parser.add_argument("--source", default=str(DEFAULT_ORTHO), help="Path to the orthomosaic GeoTIFF.")
    parser.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_WEIGHTS), help="Path to YOLO weights.")
    parser.add_argument(
        "--sam-weights",
        default=str(ROOT / "weights" / "sam_b.pt"),
        help="Path to SAM weights. Uses sam_b.pt by default.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "pipeline_outputs" / "orthomosaic"),
        help="Directory for stitched outputs and metadata.",
    )
    parser.add_argument("--tile-size", type=int, default=640, help="Tile width and height in pixels.")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.10,
        help="Tile overlap as a fraction of tile size, for example 0.10 for 10%%.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    parser.add_argument("--min-mask-area", type=int, default=50, help="Discard small mask fragments.")
    parser.add_argument(
        "--preview-max-dim",
        type=int,
        default=4096,
        help="Maximum width or height of the PNG preview.",
    )
    parser.add_argument(
        "--limit-tiles",
        type=int,
        default=0,
        help="Optional debug limit for the number of tiles to process. 0 means all tiles.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate stitched memmap files after the final outputs are written.",
    )
    return parser.parse_args()


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
        accept_y_min = y0
        accept_y_max = next_y

        for col_index, x0 in enumerate(x_starts):
            x1 = min(x0 + tile_size, width)
            next_x = x_starts[col_index + 1] if col_index + 1 < len(x_starts) else width
            accept_x_min = x0
            accept_x_max = next_x

            jobs.append(
                TileJob(
                    tile_id=tile_id,
                    x0=x0,
                    y0=y0,
                    width=x1 - x0,
                    height=y1 - y0,
                    padded_width=tile_size,
                    padded_height=tile_size,
                    accept_x_min=accept_x_min,
                    accept_x_max=accept_x_max,
                    accept_y_min=accept_y_min,
                    accept_y_max=accept_y_max,
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
    for mask in tile_mask_stack:
        cleaned = clean_mask(mask[: job.height, : job.width], min_mask_area=min_mask_area)
        if cleaned.max() == 0:
            continue

        y_slice = slice(job.y0, job.y0 + job.height)
        x_slice = slice(job.x0, job.x0 + job.width)
        stitched_mask[y_slice, x_slice] = np.maximum(stitched_mask[y_slice, x_slice], cleaned)
        stitched_counts[y_slice, x_slice] += cleaned.astype(np.uint16)
        accepted += 1
    return accepted


def write_mask_raster(source_path: Path, mask_path: Path, mask_array: np.memmap) -> None:
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="LZW")
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(np.asarray(mask_array, dtype=np.uint8), 1)


def blend_overlay(rgb_tile: np.ndarray, mask_tile: np.ndarray) -> np.ndarray:
    overlay = rgb_tile.copy()
    mask_bool = mask_tile > 0
    if not mask_bool.any():
        return overlay

    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 255
    overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.45, tint[mask_bool], 0.55, 0)
    return overlay


def write_overlay_raster(source_path: Path, overlay_path: Path, mask_array: np.memmap) -> None:
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
                mask_tile = np.asarray(mask_array[y0 : y0 + height, :], dtype=np.uint8)
                overlay = blend_overlay(tile, mask_tile)
                dst.write(np.transpose(overlay, (2, 0, 1)), window=window)


def write_preview_png(source_path: Path, preview_path: Path, mask_array: np.memmap, preview_max_dim: int) -> None:
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
    cv2.imwrite(str(preview_path), preview_overlay)


def run_pipeline(args: argparse.Namespace) -> dict:
    source = Path(args.source).resolve()
    yolo_weights = Path(args.yolo_weights).resolve()
    sam_weights = Path(args.sam_weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_inputs(source, yolo_weights, sam_weights)

    yolo_model = YOLO(str(yolo_weights))
    sam_model = SAM(str(sam_weights))

    with rasterio.open(source) as src:
        jobs = build_tile_jobs(src.width, src.height, tile_size=args.tile_size, overlap=args.overlap)
        if args.limit_tiles > 0:
            jobs = jobs[: args.limit_tiles]

        mask_memmap_path = output_dir / "stitched_mask.dat"
        count_memmap_path = output_dir / "stitched_mask_counts.dat"
        stitched_mask = np.memmap(mask_memmap_path, mode="w+", dtype=np.uint8, shape=(src.height, src.width))
        stitched_counts = np.memmap(count_memmap_path, mode="w+", dtype=np.uint16, shape=(src.height, src.width))
        stitched_mask[:] = 0
        stitched_counts[:] = 0

        total_detections = 0
        total_segmented = 0
        per_tile_summary: list[dict] = []

        for index, job in enumerate(jobs, start=1):
            rgb_tile = read_rgb_tile(src, job)
            detections = yolo_model.predict(
                source=rgb_tile,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            boxes = detections.boxes
            if boxes is None or len(boxes) == 0:
                per_tile_summary.append(
                    {
                        "tile_id": job.tile_id,
                        "x0": job.x0,
                        "y0": job.y0,
                        "detections": 0,
                        "segmented_masks": 0,
                    }
                )
                print(f"[{index}/{len(jobs)}] tile {job.tile_id}: no detections")
                continue

            raw_boxes = boxes.xyxy.cpu().numpy()
            kept_indexes = [i for i, box in enumerate(raw_boxes) if keep_detection(job, box)]
            total_detections += len(kept_indexes)

            if not kept_indexes:
                per_tile_summary.append(
                    {
                        "tile_id": job.tile_id,
                        "x0": job.x0,
                        "y0": job.y0,
                        "detections": 0,
                        "segmented_masks": 0,
                    }
                )
                print(f"[{index}/{len(jobs)}] tile {job.tile_id}: overlap-only detections skipped")
                continue

            prompt_boxes = raw_boxes[kept_indexes].tolist()
            sam_result = sam_model.predict(
                source=rgb_tile,
                bboxes=prompt_boxes,
                device=args.device,
                verbose=False,
            )[0]

            segmented_masks = 0
            if sam_result.masks is not None:
                mask_stack = sam_result.masks.data.cpu().numpy()
                segmented_masks = apply_tile_masks(
                    stitched_mask=stitched_mask,
                    stitched_counts=stitched_counts,
                    tile_mask_stack=mask_stack,
                    job=job,
                    min_mask_area=args.min_mask_area,
                )
                total_segmented += segmented_masks

            per_tile_summary.append(
                {
                    "tile_id": job.tile_id,
                    "x0": job.x0,
                    "y0": job.y0,
                    "detections": len(kept_indexes),
                    "segmented_masks": segmented_masks,
                }
            )
            print(
                f"[{index}/{len(jobs)}] tile {job.tile_id}: "
                f"{len(kept_indexes)} detections, {segmented_masks} stitched masks"
            )

        stitched_mask.flush()
        stitched_counts.flush()

    mask_tif = output_dir / "orthomosaic_mask.tif"
    overlay_tif = output_dir / "orthomosaic_overlay.tif"
    preview_png = output_dir / "orthomosaic_overlay_preview.png"
    write_mask_raster(source, mask_tif, stitched_mask)
    write_overlay_raster(source, overlay_tif, stitched_mask)
    write_preview_png(source, preview_png, stitched_mask, preview_max_dim=args.preview_max_dim)

    metadata = {
        "source": str(source),
        "yolo_weights": str(yolo_weights),
        "sam_weights": str(sam_weights),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "num_tiles": len(jobs),
        "total_detections": total_detections,
        "total_segmented_masks": total_segmented,
        "mask_raster": str(mask_tif),
        "overlay_raster": str(overlay_tif),
        "preview_png": str(preview_png),
        "intermediate_mask_memmap": str(mask_memmap_path),
        "intermediate_count_memmap": str(count_memmap_path),
        "tile_summary": per_tile_summary,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.keep_intermediates:
        del stitched_mask
        del stitched_counts
        mask_memmap_path.unlink(missing_ok=True)
        count_memmap_path.unlink(missing_ok=True)
        metadata["intermediate_mask_memmap"] = None
        metadata["intermediate_count_memmap"] = None
        (output_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def main() -> None:
    args = parse_args()
    metadata = run_pipeline(args)
    print(f"Finished orthomosaic pipeline with {metadata['total_segmented_masks']} stitched masks.")
    print(f"Overlay GeoTIFF: {metadata['overlay_raster']}")
    print(f"Preview PNG: {metadata['preview_png']}")


if __name__ == "__main__":
    main()
