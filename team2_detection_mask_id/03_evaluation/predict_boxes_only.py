from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO prediction and save images with boxes only, without labels or confidence text."
    )
    parser.add_argument("--weights", required=True, help="Path to model weights, usually best.pt.")
    parser.add_argument("--source", required=True, help="Input image file or folder.")
    parser.add_argument("--output-name", required=True, help="Output folder name under boxed_predictions.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    source = Path(args.source).resolve()
    output_root = ROOT / "boxed_predictions"

    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        show_labels=False,
        show_conf=False,
        project=str(output_root),
        name=args.output_name,
        exist_ok=True,
        verbose=True,
    )

    print(f"Saved boxed-only predictions to: {output_root / args.output_name}")


if __name__ == "__main__":
    main()
