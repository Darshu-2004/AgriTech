from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES = ROOT / "dataset" / "images" / "test"
OUTPUT_ROOT = ROOT / "boxed_predictions"

MODELS = {
    "yolov8s": ROOT / "runs" / "detect" / "runs" / "train" / "pineapple_yolov8s2" / "weights" / "best.pt",
    "yolov8m": ROOT / "runs" / "train" / "pineapple_yolov8m" / "weights" / "best.pt",
    "yolov8l": ROOT / "runs" / "train" / "pineapple_yolov8l" / "weights" / "best.pt",
    "yolov8x": ROOT / "runs" / "train" / "pineapple_yolov8x" / "weights" / "best.pt",
    "yolov8x_200e": ROOT / "runs" / "train" / "pineapple_yolov8x2" / "weights" / "best.pt",
}


def main() -> None:
    for name, weights in MODELS.items():
        if not weights.exists():
            raise FileNotFoundError(f"Missing weights for {name}: {weights}")
        print(f"Running boxed-only export for {name}...")
        model = YOLO(str(weights))
        model.predict(
            source=str(TEST_IMAGES),
            imgsz=640,
            conf=0.25,
            device="0",
            save=True,
            show_labels=False,
            show_conf=False,
            project=str(OUTPUT_ROOT),
            name=name,
            exist_ok=True,
            verbose=True,
        )
    print(f"Saved boxed-only predictions under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
