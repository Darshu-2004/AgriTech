# Stage 2 — Model Training

Trains YOLOv8 on the prepared dataset. Four model sizes are supported: s, m, l, x.

## Usage

```bash
# Train a single model (default: yolov8x, 100 epochs)
python 02_training/train_yolov8.py

# Choose model size and epochs
python 02_training/train_yolov8.py --model m --epochs 50

# Train all four sizes sequentially (PowerShell)
02_training/train_yolov8_all.ps1
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `x` | Model size: s, m, l, or x |
| `--epochs` | `100` | Number of training epochs |
| `--batch` | `-1` | Batch size (-1 = auto) |
| `--device` | `0` | GPU device or `cpu` |
| `--imgsz` | `640` | Training image size |

## Output

Weights and training logs are saved to:
```
runs/train/pineapple_yolov8{size}/
├── weights/
│   ├── best.pt     # best checkpoint
│   └── last.pt
└── results.csv     # loss and metric curves
```

## Pre-trained results

Four models have already been trained. See [`reports/FULL_MODEL_TRAINING_REPORT.md`](../reports/FULL_MODEL_TRAINING_REPORT.md) for full details.
