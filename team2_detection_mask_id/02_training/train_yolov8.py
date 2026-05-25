from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
MODEL_CHOICES = {
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
    "l": "yolov8l.pt",
    "x": "yolov8x.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model on the prepared dataset.")
    parser.add_argument("--size", choices=MODEL_CHOICES.keys(), required=True, help="Model size to train.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Training device, for example cpu, 0, or 0,1.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--project", default="runs/train", help="Output project folder.")
    parser.add_argument("--cache", default=False, action="store_true", help="Cache images for faster training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (ROOT / project_path).resolve()

    model = YOLO(MODEL_CHOICES[args.size])
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=f"pineapple_yolov8{args.size}",
        cache=args.cache,
    )


if __name__ == "__main__":
    main()
