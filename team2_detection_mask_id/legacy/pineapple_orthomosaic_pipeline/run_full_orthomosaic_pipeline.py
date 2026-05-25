from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from orthomosaic_mask_pipeline import ROOT, run_pipeline


DEFAULT_ORTHO_DIR = ROOT / "ortho"
DEFAULT_OUTPUT_DIR = ROOT / "pipeline_outputs" / "orthomosaic_full"
DEFAULT_SAM_WEIGHTS = ROOT / "weights" / "sam_b.pt"
DEFAULT_YOLO_WEIGHTS = ROOT / "weights" / "best.pt"
ORTHO_EXTENSIONS = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full orthomosaic tiling, YOLO detection, SAM segmentation, and stitched overlay pipeline."
    )
    parser.add_argument(
        "--source",
        help="Optional orthomosaic path. If omitted, the script auto-selects a TIFF from the ortho folder.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Final output directory.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size in pixels.")
    parser.add_argument("--overlap", type=float, default=0.10, help="Tile overlap fraction.")
    parser.add_argument("--min-mask-area", type=int, default=50, help="Minimum SAM mask area to keep.")
    parser.add_argument("--preview-max-dim", type=int, default=4096, help="Max preview PNG dimension.")
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate stitched memmap files instead of removing them after completion.",
    )
    return parser.parse_args()


def resolve_source(source_arg: str | None) -> Path:
    if source_arg:
        source = Path(source_arg).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Orthomosaic not found: {source}")
        return source

    ortho_candidates = sorted(
        path for path in DEFAULT_ORTHO_DIR.iterdir() if path.is_file() and path.suffix.lower() in ORTHO_EXTENSIONS
    )
    if not ortho_candidates:
        raise FileNotFoundError(f"No orthomosaic TIFF found in: {DEFAULT_ORTHO_DIR}")
    if len(ortho_candidates) > 1:
        raise RuntimeError(
            "Multiple orthomosaic TIFFs found in the ortho folder. "
            "Pass --source explicitly to choose the correct file."
        )
    return ortho_candidates[0].resolve()


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source)

    pipeline_args = SimpleNamespace(
        source=str(source),
        yolo_weights=str(DEFAULT_YOLO_WEIGHTS.resolve()),
        sam_weights=str(DEFAULT_SAM_WEIGHTS.resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        min_mask_area=args.min_mask_area,
        preview_max_dim=args.preview_max_dim,
        limit_tiles=0,
        keep_intermediates=args.keep_intermediates,
    )

    metadata = run_pipeline(pipeline_args)
    print("Full orthomosaic pipeline completed.")
    print(f"Source orthomosaic: {metadata['source']}")
    print(f"Overlay GeoTIFF: {metadata['overlay_raster']}")
    print(f"Mask GeoTIFF: {metadata['mask_raster']}")
    print(f"Preview PNG: {metadata['preview_png']}")


if __name__ == "__main__":
    main()
