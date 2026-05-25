from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "FULL_MODEL_TRAINING_REPORT.md"
DATASET_SUMMARY = ROOT / "dataset" / "split_summary.txt"
COMPARISON_REPORT = ROOT / "MODEL_COMPARISON.md"

MODEL_RUNS = {
    "s": {
        "label": "yolov8s",
        "train_dir": ROOT / "runs" / "detect" / "runs" / "train" / "pineapple_yolov8s2",
        "eval_dir": ROOT / "runs" / "evaluation" / "yolov8s",
        "pred_dir": ROOT / "predicted_test_images" / "yolov8s",
        "notes": [
            "This is the final GPU-trained `yolov8s` run used for comparison.",
            "An earlier CPU run also exists at `runs\\detect\\runs\\train\\pineapple_yolov8s` and is noted separately below.",
        ],
    },
    "m": {
        "label": "yolov8m",
        "train_dir": ROOT / "runs" / "train" / "pineapple_yolov8m",
        "eval_dir": ROOT / "runs" / "evaluation" / "yolov8m",
        "pred_dir": ROOT / "predicted_test_images" / "yolov8m",
        "notes": [],
    },
    "l": {
        "label": "yolov8l",
        "train_dir": ROOT / "runs" / "train" / "pineapple_yolov8l",
        "eval_dir": ROOT / "runs" / "evaluation" / "yolov8l",
        "pred_dir": ROOT / "predicted_test_images" / "yolov8l",
        "notes": [],
    },
    "x": {
        "label": "yolov8x",
        "train_dir": ROOT / "runs" / "train" / "pineapple_yolov8x",
        "eval_dir": ROOT / "runs" / "evaluation" / "yolov8x",
        "pred_dir": ROOT / "predicted_test_images" / "yolov8x",
        "notes": [],
    },
}


def parse_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value in (None, "", "nan", "inf", "-inf"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def fmt_seconds_as_hms(value: str | None) -> str:
    seconds = to_float(value)
    if seconds is None:
        return "n/a"
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{seconds:.4f} s ({hours:02d}:{minutes:02d}:{secs:02d})"


def read_dataset_summary() -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in DATASET_SUMMARY.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            summary[key.strip()] = value.strip()
    return summary


def collect_model_info(model_key: str, config: dict[str, object]) -> dict[str, object]:
    train_dir = config["train_dir"]
    eval_dir = config["eval_dir"]
    pred_dir = config["pred_dir"]
    args = parse_simple_yaml(train_dir / "args.yaml")
    rows = read_csv_rows(train_dir / "results.csv")
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))

    best_row = max(rows, key=lambda row: to_float(row.get("metrics/mAP50-95(B)")) or -1)
    final_row = rows[-1]

    pred_count = len(list(pred_dir.glob("*.*")))

    return {
        "key": model_key,
        "label": config["label"],
        "train_dir": train_dir,
        "eval_dir": eval_dir,
        "pred_dir": pred_dir,
        "args": args,
        "rows": rows,
        "best_row": best_row,
        "final_row": final_row,
        "metrics": metrics,
        "pred_count": pred_count,
        "notes": config["notes"],
        "weights": train_dir / "weights" / "best.pt",
        "results_plot": train_dir / "results.png",
        "train_confusion": train_dir / "confusion_matrix.png",
        "eval_confusion": eval_dir / "confusion_matrix.png",
    }


def build_report() -> str:
    dataset = read_dataset_summary()
    models = [collect_model_info(key, cfg) for key, cfg in MODEL_RUNS.items()]

    best_map95 = max(models, key=lambda item: item["metrics"]["map50_95"] or -1)
    best_precision = max(models, key=lambda item: item["metrics"]["precision"] or -1)
    best_recall = max(models, key=lambda item: item["metrics"]["recall"] or -1)
    fastest = min(models, key=lambda item: item["metrics"]["speed_inference_ms"] or float("inf"))

    lines: list[str] = [
        "# Full YOLOv8 Training Report",
        "",
        "## Dataset Preparation",
        "",
        f"- Source images exported: `{dataset.get('source_images', 'n/a')}`",
        f"- Labeled image/label pairs kept: `{dataset.get('kept_labeled_pairs', 'n/a')}`",
        f"- Unlabeled images removed: `{dataset.get('removed_unlabeled_images', 'n/a')}`",
        f"- Train split: `{dataset.get('train', 'n/a')}`",
        f"- Validation split: `{dataset.get('val', 'n/a')}`",
        f"- Test split: `{dataset.get('test', 'n/a')}`",
        f"- Dataset config: `{ROOT / 'dataset' / 'data.yaml'}`",
        "",
        "## Environment",
        "",
        "- Python: `3.14.3`",
        "- Ultralytics: `8.4.31`",
        "- PyTorch: `2.11.0+cu128`",
        "- GPU: `NVIDIA GeForce RTX 4060`",
        "- CUDA detected by PyTorch: `12.8`",
        "",
        "## Overall Summary",
        "",
        f"- Best test `mAP50-95`: `{best_map95['label']}` with `{fmt(best_map95['metrics']['map50_95'])}`",
        f"- Best test precision: `{best_precision['label']}` with `{fmt(best_precision['metrics']['precision'])}`",
        f"- Best test recall: `{best_recall['label']}` with `{fmt(best_recall['metrics']['recall'])}`",
        f"- Fastest saved-prediction inference: `{fastest['label']}` with `{fmt(fastest['metrics']['speed_inference_ms'])}` ms/image",
        "",
        "## Model Comparison Table",
        "",
        "| Model | Epochs | Batch | Device | Best Epoch | Best Val mAP50-95 | Final Val mAP50-95 | Test Precision | Test Recall | Test mAP50 | Test mAP50-95 | Test Inference ms/img | Best Weights |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for item in models:
        args = item["args"]
        best = item["best_row"]
        final = item["final_row"]
        metrics = item["metrics"]
        lines.append(
            f"| {item['label']} | {args.get('epochs', 'n/a')} | {args.get('batch', 'n/a')} | {args.get('device', 'n/a')} | "
            f"{best.get('epoch', 'n/a')} | {fmt(to_float(best.get('metrics/mAP50-95(B)')))} | {fmt(to_float(final.get('metrics/mAP50-95(B)')))} | "
            f"{fmt(metrics.get('precision'))} | {fmt(metrics.get('recall'))} | {fmt(metrics.get('map50'))} | {fmt(metrics.get('map50_95'))} | "
            f"{fmt(metrics.get('speed_inference_ms'))} | `{item['weights']}` |"
        )

    lines.extend(
        [
            "",
            "## Per-Model Detail",
            "",
        ]
    )

    for item in models:
        args = item["args"]
        best = item["best_row"]
        final = item["final_row"]
        metrics = item["metrics"]
        lines.extend(
            [
                f"### {item['label']}",
                "",
                "**Training Setup**",
                "",
                f"- Base pretrained weights: `{args.get('model', 'n/a')}`",
                f"- Epoch target: `{args.get('epochs', 'n/a')}`",
                f"- Image size: `{args.get('imgsz', 'n/a')}`",
                f"- Batch size: `{args.get('batch', 'n/a')}`",
                f"- Device used: `{args.get('device', 'n/a')}`",
                f"- Optimizer mode: `{args.get('optimizer', 'n/a')}`",
                f"- Dataset file: `{args.get('data', 'n/a')}`",
                f"- Training folder: `{item['train_dir']}`",
                f"- Evaluation folder: `{item['eval_dir']}`",
                f"- Prediction image folder: `{item['pred_dir']}`",
                f"- Best weights: `{item['weights']}`",
                f"- Training curve image: `{item['results_plot']}`",
                f"- Training confusion matrix: `{item['train_confusion']}`",
                f"- Evaluation confusion matrix: `{item['eval_confusion']}`",
                "",
                "**Best Validation Epoch**",
                "",
                f"- Best epoch by validation `mAP50-95`: `{best.get('epoch', 'n/a')}`",
                f"- Cumulative training time at best epoch: `{fmt_seconds_as_hms(best.get('time'))}`",
                f"- Train box loss at best epoch: `{fmt(to_float(best.get('train/box_loss')))} `",
                f"- Train cls loss at best epoch: `{fmt(to_float(best.get('train/cls_loss')))} `",
                f"- Train dfl loss at best epoch: `{fmt(to_float(best.get('train/dfl_loss')))} `",
                f"- Validation precision at best epoch: `{fmt(to_float(best.get('metrics/precision(B)')))} `",
                f"- Validation recall at best epoch: `{fmt(to_float(best.get('metrics/recall(B)')))} `",
                f"- Validation mAP50 at best epoch: `{fmt(to_float(best.get('metrics/mAP50(B)')))} `",
                f"- Validation mAP50-95 at best epoch: `{fmt(to_float(best.get('metrics/mAP50-95(B)')))} `",
                f"- Validation box loss at best epoch: `{fmt(to_float(best.get('val/box_loss')))} `",
                f"- Validation cls loss at best epoch: `{fmt(to_float(best.get('val/cls_loss')))} `",
                f"- Validation dfl loss at best epoch: `{fmt(to_float(best.get('val/dfl_loss')))} `",
                "",
                "**Final Training Epoch**",
                "",
                f"- Final epoch: `{final.get('epoch', 'n/a')}`",
                f"- Total recorded training time: `{fmt_seconds_as_hms(final.get('time'))}`",
                f"- Final train box loss: `{fmt(to_float(final.get('train/box_loss')))} `",
                f"- Final train cls loss: `{fmt(to_float(final.get('train/cls_loss')))} `",
                f"- Final train dfl loss: `{fmt(to_float(final.get('train/dfl_loss')))} `",
                f"- Final validation precision: `{fmt(to_float(final.get('metrics/precision(B)')))} `",
                f"- Final validation recall: `{fmt(to_float(final.get('metrics/recall(B)')))} `",
                f"- Final validation mAP50: `{fmt(to_float(final.get('metrics/mAP50(B)')))} `",
                f"- Final validation mAP50-95: `{fmt(to_float(final.get('metrics/mAP50-95(B)')))} `",
                f"- Final validation box loss: `{fmt(to_float(final.get('val/box_loss')))} `",
                f"- Final validation cls loss: `{fmt(to_float(final.get('val/cls_loss')))} `",
                f"- Final validation dfl loss: `{fmt(to_float(final.get('val/dfl_loss')))} `",
                "",
                "**Test Set Evaluation**",
                "",
                f"- Test precision: `{fmt(metrics.get('precision'))}`",
                f"- Test recall: `{fmt(metrics.get('recall'))}`",
                f"- Test mAP50: `{fmt(metrics.get('map50'))}`",
                f"- Test mAP50-95: `{fmt(metrics.get('map50_95'))}`",
                f"- Test fitness: `{fmt(metrics.get('fitness'))}`",
                f"- Test preprocess speed: `{fmt(metrics.get('speed_preprocess_ms'))}` ms/image",
                f"- Test inference speed: `{fmt(metrics.get('speed_inference_ms'))}` ms/image",
                f"- Test postprocess speed: `{fmt(metrics.get('speed_postprocess_ms'))}` ms/image",
                f"- Boxed prediction images saved: `{item['pred_count']}` files",
                "",
                "**Artifacts and Analysis Notes**",
                "",
            ]
        )
        if item["notes"]:
            for note in item["notes"]:
                lines.append(f"- {note}")
        else:
            lines.append("- No extra run notes for this model.")
        lines.extend(
            [
                "",
            ]
        )

    lines.extend(
        [
            "## Additional Notes",
            "",
            f"- Short comparison report: `{COMPARISON_REPORT}`",
            "- The `yolov8s` family has two run folders because an earlier CPU attempt existed before the final GPU-standardized run.",
            "- Final comparison and prediction outputs were produced from the GPU-trained runs.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Saved full training report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
