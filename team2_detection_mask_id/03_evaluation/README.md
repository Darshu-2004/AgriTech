# Stage 3 — Evaluation

Evaluates all trained models on the test split and generates a comparison report.

## Usage

```bash
# Evaluate all four models and write MODEL_COMPARISON.md
python 03_evaluation/evaluate_and_report.py

# Evaluate a single model run with detailed output
python 03_evaluation/evaluate_single_run.py --run pineapple_yolov8x

# Save predicted bounding-box images for one model
python 03_evaluation/predict_boxes_only.py --model x

# Export boxed predictions for all models
python 03_evaluation/export_all_test_boxes_only.py
```

## Output

```
runs/evaluation/{model}/
├── metrics.json        # precision, recall, mAP50, mAP50-95, speed
└── predictions.json    # per-image prediction records

predicted_test_images/{model}/
└── *.jpg               # test images with drawn bounding boxes

reports/MODEL_COMPARISON.md   # generated summary table
```

## Results (test set)

| Model | mAP50-95 | Precision | Recall | ms/img |
|-------|----------|-----------|--------|--------|
| yolov8s | 0.8365 | 0.9509 | 0.9854 | 29.2 |
| yolov8m | 0.8475 | 0.9546 | 0.9763 | 27.2 |
| yolov8l | 0.8333 | 0.9609 | 0.9682 | 25.6 |
| yolov8x | 0.8370 | 0.9540 | 0.9764 | 42.2 |
