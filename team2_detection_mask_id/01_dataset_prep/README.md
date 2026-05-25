# Stage 1 — Dataset Preparation

Splits the labelled source images from `drone_d_img/` into train / val / test sets and writes them to `dataset/`.

## Usage

```bash
python 01_dataset_prep/prepare_yolov8_dataset.py
```

## What it does

1. Scans `drone_d_img/images/` for images that have a matching label file in `drone_d_img/labels/train/`
2. Shuffles the matched pairs with a fixed seed (42) for reproducibility
3. Splits 80 % train / 10 % val / 10 % test
4. Copies images and labels into `dataset/images/{split}/` and `dataset/labels/{split}/`
5. Writes `dataset/data.yaml` (YOLO config) and `dataset/split_summary.txt`

## Output

```
dataset/
├── data.yaml
├── split_summary.txt
├── images/
│   ├── train/   (~155 images)
│   ├── val/     (~19 images)
│   └── test/    (~19 images)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Notes

- Source images must be placed in `drone_d_img/images/` with labels in `drone_d_img/labels/train/`
- Re-running will reset the dataset folder
