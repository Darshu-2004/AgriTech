from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
RUNS_ROOT = ROOT / "runs"
TRAIN_ROOT = RUNS_ROOT / "train"
EVAL_ROOT = RUNS_ROOT / "evaluation"
PRED_ROOT = ROOT / "predicted_test_images"
REPORT_PATH = ROOT / "MODEL_COMPARISON.md"
MODEL_SIZES = ("s", "m", "l", "x")


def metric_value(results_dict: dict, *keys: str) -> float | None:
    for key in keys:
        value = results_dict.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def load_model_path(size: str) -> Path:
    candidates = sorted(
        ROOT.glob(f"runs/**/pineapple_yolov8{size}*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Trained model not found for yolov8{size} under {ROOT / 'runs'}")
    return candidates[0]


def evaluate_model(size: str) -> dict:
    model_path = load_model_path(size)
    eval_dir = EVAL_ROOT / f"yolov8{size}"
    pred_dir = PRED_ROOT / f"yolov8{size}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    val_results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        project=str(EVAL_ROOT),
        name=f"yolov8{size}",
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=True,
    )
    pred_results = model.predict(
        source=str(ROOT / "dataset" / "images" / "test"),
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(PRED_ROOT),
        name=f"yolov8{size}",
        exist_ok=True,
        verbose=True,
    )

    prediction_speeds = [getattr(result, "speed", {}) or {} for result in pred_results]
    pred_preprocess = [item.get("preprocess") for item in prediction_speeds if item.get("preprocess") is not None]
    pred_inference = [item.get("inference") for item in prediction_speeds if item.get("inference") is not None]
    pred_postprocess = [item.get("postprocess") for item in prediction_speeds if item.get("postprocess") is not None]

    def avg(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    results_dict = dict(getattr(val_results, "results_dict", {}) or {})
    metrics = {
        "size": size,
        "model_path": str(model_path),
        "prediction_count": len(pred_results),
        "precision": metric_value(results_dict, "metrics/precision(B)", "metrics/precision"),
        "recall": metric_value(results_dict, "metrics/recall(B)", "metrics/recall"),
        "map50": metric_value(results_dict, "metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": metric_value(results_dict, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "fitness": metric_value(results_dict, "fitness"),
        "speed_preprocess_ms": metric_value(results_dict, "speed/preprocess") or avg(pred_preprocess),
        "speed_inference_ms": metric_value(results_dict, "speed/inference") or avg(pred_inference),
        "speed_postprocess_ms": metric_value(results_dict, "speed/postprocess") or avg(pred_postprocess),
        "eval_dir": str(eval_dir),
        "pred_dir": str(pred_dir),
        "results_dict": results_dict,
    }

    metrics_path = eval_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_report(metrics_by_model: list[dict]) -> None:
    best_map = max(metrics_by_model, key=lambda item: item["map50_95"] if item["map50_95"] is not None else -1)
    best_map50 = max(metrics_by_model, key=lambda item: item["map50"] if item["map50"] is not None else -1)
    fastest = min(
        metrics_by_model,
        key=lambda item: item["speed_inference_ms"] if item["speed_inference_ms"] is not None else float("inf"),
    )

    lines = [
        "# YOLOv8 Model Comparison",
        "",
        "## Summary",
        "",
        f"- Best mAP50-95: `yolov8{best_map['size']}` with `{fmt(best_map['map50_95'])}`",
        f"- Best mAP50: `yolov8{best_map50['size']}` with `{fmt(best_map50['map50'])}`",
        f"- Fastest inference on test split: `yolov8{fastest['size']}` with `{fmt(fastest['speed_inference_ms'])}` ms/image",
        "",
        "## Metrics Table",
        "",
        "| Model | Precision | Recall | mAP50 | mAP50-95 | Fitness | Inference ms/img | Predictions Folder | Evaluation Folder |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for item in metrics_by_model:
        lines.append(
            f"| yolov8{item['size']} | {fmt(item['precision'])} | {fmt(item['recall'])} | "
            f"{fmt(item['map50'])} | {fmt(item['map50_95'])} | {fmt(item['fitness'])} | "
            f"{fmt(item['speed_inference_ms'])} | `{item['pred_dir']}` | `{item['eval_dir']}` |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Metrics are computed on the isolated `test` split.",
            "- Boxed prediction images are saved separately for each model under `predicted_test_images/`.",
            "- Raw per-model metrics are also saved as `metrics.json` inside each evaluation folder.",
            "",
            "## Per-Model Artifacts",
            "",
        ]
    )

    for item in metrics_by_model:
        lines.extend(
            [
                f"### yolov8{item['size']}",
                "",
                f"- Weights: `{item['model_path']}`",
                f"- Prediction images: `{item['pred_dir']}`",
                f"- Evaluation outputs: `{item['eval_dir']}`",
                f"- Precision: `{fmt(item['precision'])}`",
                f"- Recall: `{fmt(item['recall'])}`",
                f"- mAP50: `{fmt(item['map50'])}`",
                f"- mAP50-95: `{fmt(item['map50_95'])}`",
                f"- Inference speed: `{fmt(item['speed_inference_ms'])}` ms/image",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    metrics_by_model = [evaluate_model(size) for size in MODEL_SIZES]
    write_report(metrics_by_model)
    print(f"Saved comparison report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
