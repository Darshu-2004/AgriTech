"""
Master pipeline runner — pineapple detection and plant ID assignment.

Runs Stage 1 (orthomosaic tiling + YOLO + SAM segmentation) then
Stage 2 (centroid clustering + plant ID export) and writes all results
into a single timestamped output directory.

Usage
-----
  # Full run (auto-detects TIFF in ortho/)
  python run_pipeline.py

  # Explicit orthomosaic
  python run_pipeline.py --ortho ortho/ggs-orthophoto.tif

  # Custom output dir
  python run_pipeline.py --output-dir my_results/run_01

  # Skip Stage 1 if the mask already exists
  python run_pipeline.py --skip-ortho --output-dir my_results/run_01

  # CPU-only
  python run_pipeline.py --device cpu
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ORTHO_DIR        = ROOT / "ortho"
YOLO_WEIGHTS     = ROOT / "runs" / "train" / "pineapple_yolov8x2" / "weights" / "best.pt"
SAM_WEIGHTS      = ROOT / "sam_b.pt"
STAGE1_SCRIPT    = ROOT / "04_orthomosaic"  / "run_full_orthomosaic_pipeline.py"
STAGE2_SCRIPT    = ROOT / "05_plant_ids"    / "run_plant_id_pipeline.py"
ORTHO_EXTENSIONS = {".tif", ".tiff"}


def _resolve_ortho(ortho_arg: str | None) -> Path:
    if ortho_arg:
        p = Path(ortho_arg).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Orthomosaic not found: {p}")
        return p
    candidates = sorted(
        p for p in ORTHO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ORTHO_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No TIFF found in {ORTHO_DIR}. Pass --ortho explicitly.")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RuntimeError(f"Multiple TIFFs in ortho/: {names}. Pass --ortho explicitly.")
    return candidates[0].resolve()


def _run(cmd: list[str], label: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {label}\n{bar}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\nPipeline aborted: {label} failed (exit {result.returncode})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full pineapple detection and plant ID pipeline."
    )
    p.add_argument("--ortho",            default=None,  help="Path to input orthomosaic GeoTIFF.")
    p.add_argument("--output-dir",       default=None,  help="Root output directory. Default: outputs/run_TIMESTAMP.")
    p.add_argument("--device",           default="0",   help="GPU device index or 'cpu'. Default: 0.")
    p.add_argument("--conf",             type=float, default=0.25, help="YOLO confidence threshold.")
    p.add_argument("--tile-size",        type=int,   default=640,  help="Tile size in pixels.")
    p.add_argument("--overlap",          type=float, default=0.10, help="Tile overlap fraction.")
    p.add_argument("--min-mask-area",    type=int,   default=50,   help="Minimum SAM mask area (px).")
    p.add_argument("--min-cluster-size", type=int,   default=25,   help="HDBSCAN min cluster size.")
    p.add_argument("--skip-ortho",       action="store_true",
                   help="Skip Stage 1 — use existing mask from --output-dir/01_orthomosaic/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ortho = _resolve_ortho(args.ortho)

    if args.output_dir:
        out_root = Path(args.output_dir).resolve()
    else:
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = ROOT / "outputs" / f"run_{stamp}"

    stage1_out = out_root / "01_orthomosaic"
    stage2_out = out_root / "02_plant_ids"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nPineapple Detection Pipeline")
    print(f"Orthomosaic : {ortho}")
    print(f"Output root : {out_root}")
    print(f"Device      : {args.device}")

    if not args.skip_ortho:
        _run([
            sys.executable, str(STAGE1_SCRIPT),
            "--source",       str(ortho),
            "--output-dir",   str(stage1_out),
            "--device",       args.device,
            "--conf",         str(args.conf),
            "--tile-size",    str(args.tile_size),
            "--overlap",      str(args.overlap),
            "--min-mask-area", str(args.min_mask_area),
        ], "Stage 1 — Orthomosaic tiling + YOLO detection + SAM segmentation")
    else:
        print("\nStage 1 skipped (--skip-ortho). Using existing mask.")

    run_summary = stage1_out / "run_summary.json"
    mask_tif    = stage1_out / "orthomosaic_mask.tif"
    if not run_summary.exists():
        sys.exit(f"\nStage 2 cannot start: run_summary.json not found at {run_summary}")
    if not mask_tif.exists():
        sys.exit(f"\nStage 2 cannot start: orthomosaic_mask.tif not found at {mask_tif}")

    _run([
        sys.executable, str(STAGE2_SCRIPT),
        "--source",           str(ortho),
        "--run-summary",      str(run_summary),
        "--mask",             str(mask_tif),
        "--output-dir",       str(stage2_out),
        "--min-cluster-size", str(args.min_cluster_size),
    ], "Stage 2 — Plant centroid clustering + ID assignment")

    print(f"\n{'='*60}")
    print("  Pipeline complete")
    print(f"{'='*60}")
    print(f"Results      : {out_root}")
    print(f"Mask GeoTIFF : {stage1_out / 'orthomosaic_mask.tif'}")
    print(f"Overlay TIFF : {stage1_out / 'orthomosaic_overlay.tif'}")
    print(f"Plant IDs    : {stage2_out / 'plants.geojson'}")
    print(f"Viewer       : {stage2_out / 'viewer.html'}")


if __name__ == "__main__":
    main()
