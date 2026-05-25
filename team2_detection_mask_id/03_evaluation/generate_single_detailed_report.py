from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a detailed markdown report for one training run.")
    parser.add_argument("--train-dir", required=True, help="Training run directory.")
    parser.add_argument("--eval-dir", required=True, help="Evaluation directory.")
    parser.add_argument("--pred-dir", required=True, help="Prediction image directory.")
    parser.add_argument("--title", required=True, help="Report title.")
    parser.add_argument("--output", required=True, help="Markdown output path.")
    return parser.parse_args()


def parse_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or ":" not in line:
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


def fmt_seconds(value: str | None) -> str:
    seconds = to_float(value)
    if seconds is None:
        return "n/a"
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{seconds:.4f} s ({h:02d}:{m:02d}:{s:02d})"


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir).resolve()
    eval_dir = Path(args.eval_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    output = Path(args.output).resolve()

    cfg = parse_simple_yaml(train_dir / "args.yaml")
    rows = read_csv_rows(train_dir / "results.csv")
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    best = max(rows, key=lambda row: to_float(row.get("metrics/mAP50-95(B)")) or -1)
    final = rows[-1]

    lines = [
        f"# {args.title}",
        "",
        "## Training Setup",
        "",
        f"- Base model: `{cfg.get('model', 'n/a')}`",
        f"- Epochs: `{cfg.get('epochs', 'n/a')}`",
        f"- Batch size: `{cfg.get('batch', 'n/a')}`",
        f"- Image size: `{cfg.get('imgsz', 'n/a')}`",
        f"- Device: `{cfg.get('device', 'n/a')}`",
        f"- Dataset: `{cfg.get('data', 'n/a')}`",
        f"- Optimizer mode: `{cfg.get('optimizer', 'n/a')}`",
        f"- Training folder: `{train_dir}`",
        f"- Evaluation folder: `{eval_dir}`",
        f"- Prediction image folder: `{pred_dir}`",
        f"- Best weights: `{train_dir / 'weights' / 'best.pt'}`",
        "",
        "## Training Summary",
        "",
        f"- Recorded epochs in `results.csv`: `{len(rows)}`",
        f"- Best validation epoch by mAP50-95: `{best.get('epoch', 'n/a')}`",
        f"- Time at best epoch: `{fmt_seconds(best.get('time'))}`",
        f"- Total recorded training time: `{fmt_seconds(final.get('time'))}`",
        "",
        "## Best Epoch Metrics",
        "",
        f"- Train box loss: `{fmt(to_float(best.get('train/box_loss')))} `",
        f"- Train cls loss: `{fmt(to_float(best.get('train/cls_loss')))} `",
        f"- Train dfl loss: `{fmt(to_float(best.get('train/dfl_loss')))} `",
        f"- Val precision: `{fmt(to_float(best.get('metrics/precision(B)')))} `",
        f"- Val recall: `{fmt(to_float(best.get('metrics/recall(B)')))} `",
        f"- Val mAP50: `{fmt(to_float(best.get('metrics/mAP50(B)')))} `",
        f"- Val mAP50-95: `{fmt(to_float(best.get('metrics/mAP50-95(B)')))} `",
        f"- Val box loss: `{fmt(to_float(best.get('val/box_loss')))} `",
        f"- Val cls loss: `{fmt(to_float(best.get('val/cls_loss')))} `",
        f"- Val dfl loss: `{fmt(to_float(best.get('val/dfl_loss')))} `",
        "",
        "## Final Epoch Metrics",
        "",
        f"- Final epoch: `{final.get('epoch', 'n/a')}`",
        f"- Final train box loss: `{fmt(to_float(final.get('train/box_loss')))} `",
        f"- Final train cls loss: `{fmt(to_float(final.get('train/cls_loss')))} `",
        f"- Final train dfl loss: `{fmt(to_float(final.get('train/dfl_loss')))} `",
        f"- Final val precision: `{fmt(to_float(final.get('metrics/precision(B)')))} `",
        f"- Final val recall: `{fmt(to_float(final.get('metrics/recall(B)')))} `",
        f"- Final val mAP50: `{fmt(to_float(final.get('metrics/mAP50(B)')))} `",
        f"- Final val mAP50-95: `{fmt(to_float(final.get('metrics/mAP50-95(B)')))} `",
        "",
        "## Test Set Evaluation",
        "",
        f"- Precision: `{fmt(metrics.get('precision'))}`",
        f"- Recall: `{fmt(metrics.get('recall'))}`",
        f"- mAP50: `{fmt(metrics.get('map50'))}`",
        f"- mAP50-95: `{fmt(metrics.get('map50_95'))}`",
        f"- Fitness: `{fmt(metrics.get('fitness'))}`",
        f"- Preprocess speed: `{fmt(metrics.get('speed_preprocess_ms'))}` ms/image",
        f"- Inference speed: `{fmt(metrics.get('speed_inference_ms'))}` ms/image",
        f"- Postprocess speed: `{fmt(metrics.get('speed_postprocess_ms'))}` ms/image",
        f"- Predicted test images saved: `{metrics.get('prediction_count', 'n/a')}`",
        "",
        "## Artifacts",
        "",
        f"- Training curves: `{train_dir / 'results.png'}`",
        f"- Training confusion matrix: `{train_dir / 'confusion_matrix.png'}`",
        f"- Evaluation confusion matrix: `{eval_dir / 'confusion_matrix.png'}`",
        f"- Precision-recall curve: `{eval_dir / 'PR_curve.png'}`",
        f"- Precision curve: `{eval_dir / 'P_curve.png'}`",
        f"- Recall curve: `{eval_dir / 'R_curve.png'}`",
        f"- F1 curve: `{eval_dir / 'F1_curve.png'}`",
        "",
    ]

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved detailed report to: {output}")


if __name__ == "__main__":
    main()
