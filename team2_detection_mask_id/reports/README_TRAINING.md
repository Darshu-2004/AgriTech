# YOLOv8 Training Setup

This workspace has been prepared for YOLOv8 object detection training.

## Dataset layout

- `dataset/images/train`
- `dataset/images/val`
- `dataset/images/test`
- `dataset/labels/train`
- `dataset/labels/val`
- `dataset/labels/test`
- `dataset/data.yaml`

Only images that have a matching label file are included in the dataset.

## Prepare dataset

```powershell
python .\prepare_yolov8_dataset.py
```

## Train one model

```powershell
python .\train_yolov8.py --size s --epochs 100 --imgsz 640 --batch 16 --device cpu
python .\train_yolov8.py --size m --epochs 100 --imgsz 640 --batch 16 --device cpu
python .\train_yolov8.py --size l --epochs 100 --imgsz 640 --batch 8 --device cpu
python .\train_yolov8.py --size x --epochs 100 --imgsz 640 --batch 8 --device cpu
```

Replace `--device cpu` with `--device 0` if you want to train on your first CUDA GPU.

## Train all four sizes

```powershell
.\train_yolov8_all.ps1
```

## Outputs

Training runs are written under `runs/train`.

## Evaluate and compare all trained models

```powershell
python .\evaluate_and_report.py
```

This creates:

- `predicted_test_images\yolov8s`
- `predicted_test_images\yolov8m`
- `predicted_test_images\yolov8l`
- `predicted_test_images\yolov8x`
- `MODEL_COMPARISON.md`
