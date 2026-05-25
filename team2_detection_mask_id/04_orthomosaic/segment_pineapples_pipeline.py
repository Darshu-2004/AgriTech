from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLO


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YOLO_WEIGHTS = ROOT / "runs" / "train" / "pineapple_yolov8x2" / "weights" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DetectionRecord:
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    polygon: list[list[float]]
    mask_area_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 3-stage pineapple tree pipeline: YOLOv8 detection, "
            "SAM 2 prompted segmentation, and polygon export."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an input image or a directory of images.",
    )
    parser.add_argument(
        "--yolo-weights",
        default=str(DEFAULT_YOLO_WEIGHTS),
        help="Path to YOLOv8 weights. Defaults to the latest trained yolov8x checkpoint.",
    )
    parser.add_argument(
        "--sam-weights",
        required=True,
        help="Path to SAM 2 weights, for example sam2_b.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "pipeline_outputs"),
        help="Directory where JSON, masks, and overlays will be written.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=50,
        help="Discard mask fragments smaller than this many pixels.",
    )
    parser.add_argument(
        "--contour-epsilon",
        type=float,
        default=2.0,
        help="Douglas-Peucker simplification epsilon in pixels for exported polygons.",
    )
    parser.add_argument(
        "--no-preview-png",
        action="store_true",
        help="Disable saving PNG preview outputs.",
    )
    return parser.parse_args()


def validate_weights(yolo_weights: Path, sam_weights: Path) -> None:
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
    if not sam_weights.exists():
        raise FileNotFoundError(f"SAM weights not found: {sam_weights}")


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {source}")
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(f"Source path not found: {source}")

    images = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No supported images found in directory: {source}")
    return images


def contour_to_polygon(contour: np.ndarray, epsilon: float) -> list[list[float]]:
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
    polygon = simplified.reshape(-1, 2).astype(float).tolist()
    if len(polygon) >= 3 and polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon


def mask_to_polygon(mask: np.ndarray, min_mask_area: int, contour_epsilon: float) -> tuple[list[list[float]], int]:
    binary_mask = (mask > 0).astype(np.uint8)
    area = int(binary_mask.sum())
    if area < min_mask_area:
        return [], area

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], area

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_mask_area:
        return [], area

    polygon = contour_to_polygon(contour, contour_epsilon)
    return polygon, area


def build_detection_records(
    detection_result,
    segmentation_result,
    min_mask_area: int,
    contour_epsilon: float,
) -> list[DetectionRecord]:
    if detection_result.boxes is None or len(detection_result.boxes) == 0:
        return []
    if segmentation_result.masks is None:
        return []

    boxes = detection_result.boxes
    masks = segmentation_result.masks.data.cpu().numpy()
    xyxy_boxes = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)

    names = detection_result.names
    mask_count = min(len(masks), len(xyxy_boxes), len(confidences), len(class_ids))
    records: list[DetectionRecord] = []
    for index in range(mask_count):
        mask = masks[index]
        polygon, area = mask_to_polygon(mask, min_mask_area=min_mask_area, contour_epsilon=contour_epsilon)
        if not polygon:
            continue

        class_id = int(class_ids[index])
        if isinstance(names, dict):
            class_name = names.get(class_id, str(class_id))
        elif 0 <= class_id < len(names):
            class_name = str(names[class_id])
        else:
            class_name = str(class_id)
        records.append(
            DetectionRecord(
                detection_id=index,
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidences[index]),
                bbox_xyxy=[float(value) for value in xyxy_boxes[index].tolist()],
                polygon=polygon,
                mask_area_pixels=area,
            )
        )
    return records


def save_overlay(image: np.ndarray, masks: np.ndarray, destination: Path) -> None:
    overlay = image.copy()
    tint = np.zeros_like(image)
    tint[:, :, 1] = 255

    combined_mask = (masks.sum(axis=0) > 0).astype(np.uint8)
    overlay[combined_mask.astype(bool)] = cv2.addWeighted(
        image[combined_mask.astype(bool)],
        0.4,
        tint[combined_mask.astype(bool)],
        0.6,
        0,
    )
    cv2.imwrite(str(destination), overlay)


def save_binary_mask(masks: np.ndarray, destination: Path) -> None:
    combined_mask = ((masks.sum(axis=0) > 0) * 255).astype(np.uint8)
    cv2.imwrite(str(destination), combined_mask)


def save_colored_mask(masks: np.ndarray, destination: Path) -> None:
    combined_mask = (masks.sum(axis=0) > 0).astype(np.uint8)
    colored = np.zeros((combined_mask.shape[0], combined_mask.shape[1], 3), dtype=np.uint8)
    colored[combined_mask.astype(bool)] = (0, 255, 0)
    cv2.imwrite(str(destination), colored)


def process_image(
    image_path: Path,
    yolo_model: YOLO,
    sam_model: SAM,
    output_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
    min_mask_area: int,
    contour_epsilon: float,
    save_preview_png: bool,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    detection_results = yolo_model.predict(
        source=image,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    detection_result = detection_results[0]

    if detection_result.boxes is None or len(detection_result.boxes) == 0:
        payload = {
            "image": image_path.name,
            "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "detections": [],
        }
        json_path = output_dir / f"{image_path.stem}.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    bboxes = detection_result.boxes.xyxy.cpu().numpy().tolist()
    segmentation_results = sam_model.predict(
        source=image,
        bboxes=bboxes,
        device=device,
        verbose=False,
    )
    segmentation_result = segmentation_results[0]
    records = build_detection_records(
        detection_result=detection_result,
        segmentation_result=segmentation_result,
        min_mask_area=min_mask_area,
        contour_epsilon=contour_epsilon,
    )

    payload = {
        "image": image_path.name,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "detections": [
            {
                "detection_id": record.detection_id,
                "class_id": record.class_id,
                "class_name": record.class_name,
                "confidence": round(record.confidence, 6),
                "bbox_xyxy": [round(value, 3) for value in record.bbox_xyxy],
                "mask_area_pixels": record.mask_area_pixels,
                "polygon": [[round(x, 3), round(y, 3)] for x, y in record.polygon],
            }
            for record in records
        ],
    }

    json_path = output_dir / f"{image_path.stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if save_preview_png and segmentation_result.masks is not None:
        masks = segmentation_result.masks.data.cpu().numpy()
        save_overlay(image, masks, output_dir / f"{image_path.stem}_overlay.png")
        save_binary_mask(masks, output_dir / f"{image_path.stem}_mask.png")
        save_colored_mask(masks, output_dir / f"{image_path.stem}_mask_preview.png")

    return payload


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    yolo_weights = Path(args.yolo_weights).resolve()
    sam_weights = Path(args.sam_weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_weights(yolo_weights, sam_weights)
    image_paths = collect_images(source)

    yolo_model = YOLO(str(yolo_weights))
    sam_model = SAM(str(sam_weights))

    summary: list[dict] = []
    for image_path in image_paths:
        payload = process_image(
            image_path=image_path,
            yolo_model=yolo_model,
            sam_model=sam_model,
            output_dir=output_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            min_mask_area=args.min_mask_area,
            contour_epsilon=args.contour_epsilon,
            save_preview_png=not args.no_preview_png,
        )
        summary.append(
            {
                "image": payload["image"],
                "num_detections": len(payload["detections"]),
            }
        )
        print(f"{image_path.name}: exported {len(payload['detections'])} polygon(s)")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved pipeline outputs to: {output_dir}")


if __name__ == "__main__":
    main()
