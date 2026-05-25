from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "data.yaml"


def avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one trained YOLO run and write a report.")
    parser.add_argument("--weights", required=True, help="Path to trained best.pt weights.")
    parser.add_argument("--label", required=True, help="Human-readable label for the model run.")
    parser.add_argument("--eval-name", required=True, help="Folder name under runs/evaluation.")
    parser.add_argument("--pred-name", required=True, help="Folder name under predicted_test_images.")
    parser.add_argument("--report", required=True, help="Markdown report output path.")
    return parser.parse_args()


def fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def metric_value(results_dict: dict, *keys: str) -> float | None:
    for key in keys:
        value = results_dict.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    report_path = Path(args.report).resolve()
    eval_dir = (ROOT / "runs" / "evaluation" / args.eval_name).resolve()
    pred_dir = (ROOT / "predicted_test_images" / args.pred_name).resolve()
    train_dir = weights.parent.parent

    model = YOLO(str(weights))
    val_results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        project=str(ROOT / "runs" / "evaluation"),
        name=args.eval_name,
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
        project=str(ROOT / "predicted_test_images"),
        name=args.pred_name,
        exist_ok=True,
        verbose=True,
    )

    results_dict = dict(getattr(val_results, "results_dict", {}) or {})
    prediction_speeds = [getattr(result, "speed", {}) or {} for result in pred_results]
    pred_preprocess = [item.get("preprocess") for item in prediction_speeds if item.get("preprocess") is not None]
    pred_inference = [item.get("inference") for item in prediction_speeds if item.get("inference") is not None]
    pred_postprocess = [item.get("postprocess") for item in prediction_speeds if item.get("postprocess") is not None]

    metrics = {
        "precision": metric_value(results_dict, "metrics/precision(B)", "metrics/precision"),
        "recall": metric_value(results_dict, "metrics/recall(B)", "metrics/recall"),
        "map50": metric_value(results_dict, "metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": metric_value(results_dict, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "fitness": metric_value(results_dict, "fitness"),
        "speed_preprocess_ms": avg(pred_preprocess),
        "speed_inference_ms": avg(pred_inference),
        "speed_postprocess_ms": avg(pred_postprocess),
        "prediction_count": len(pred_results),
    }

    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        f"# {args.label} Report",
        "",
        "## Run",
        "",
        f"- Weights: `{weights}`",
        f"- Training folder: `{train_dir}`",
        f"- Evaluation folder: `{eval_dir}`",
        f"- Prediction image folder: `{pred_dir}`",
        "",
        "## Test Metrics",
        "",
        f"- Precision: `{fmt(metrics['precision'])}`",
        f"- Recall: `{fmt(metrics['recall'])}`",
        f"- mAP50: `{fmt(metrics['map50'])}`",
        f"- mAP50-95: `{fmt(metrics['map50_95'])}`",
        f"- Fitness: `{fmt(metrics['fitness'])}`",
        f"- Preprocess speed: `{fmt(metrics['speed_preprocess_ms'])}` ms/image",
        f"- Inference speed: `{fmt(metrics['speed_inference_ms'])}` ms/image",
        f"- Postprocess speed: `{fmt(metrics['speed_postprocess_ms'])}` ms/image",
        f"- Boxed prediction images saved: `{metrics['prediction_count']}`",
        "",
        "## Artifacts",
        "",
        f"- Confusion matrix: `{eval_dir / 'confusion_matrix.png'}`",
        f"- PR curve: `{eval_dir / 'PR_curve.png'}`",
        f"- P curve: `{eval_dir / 'P_curve.png'}`",
        f"- R curve: `{eval_dir / 'R_curve.png'}`",
        f"- F1 curve: `{eval_dir / 'F1_curve.png'}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
