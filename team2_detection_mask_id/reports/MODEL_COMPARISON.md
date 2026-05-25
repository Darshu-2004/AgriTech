# YOLOv8 Model Comparison

## Summary

- Best mAP50-95: `yolov8m` with `0.8475`
- Best mAP50: `yolov8s` with `0.9906`
- Fastest inference on test split: `yolov8l` with `25.6052` ms/image

## Metrics Table

| Model | Precision | Recall | mAP50 | mAP50-95 | Fitness | Inference ms/img | Predictions Folder | Evaluation Folder |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| yolov8s | 0.9509 | 0.9854 | 0.9906 | 0.8365 | 0.8365 | 29.1845 | `C:\drone_big\predicted_test_images\yolov8s` | `C:\drone_big\runs\evaluation\yolov8s` |
| yolov8m | 0.9546 | 0.9763 | 0.9897 | 0.8475 | 0.8475 | 27.1604 | `C:\drone_big\predicted_test_images\yolov8m` | `C:\drone_big\runs\evaluation\yolov8m` |
| yolov8l | 0.9609 | 0.9682 | 0.9903 | 0.8333 | 0.8333 | 25.6052 | `C:\drone_big\predicted_test_images\yolov8l` | `C:\drone_big\runs\evaluation\yolov8l` |
| yolov8x | 0.9540 | 0.9764 | 0.9901 | 0.8370 | 0.8370 | 42.2261 | `C:\drone_big\predicted_test_images\yolov8x` | `C:\drone_big\runs\evaluation\yolov8x` |

## Notes

- Metrics are computed on the isolated `test` split.
- Boxed prediction images are saved separately for each model under `predicted_test_images/`.
- Raw per-model metrics are also saved as `metrics.json` inside each evaluation folder.

## Per-Model Artifacts

### yolov8s

- Weights: `C:\drone_big\runs\detect\runs\train\pineapple_yolov8s\weights\best.pt`
- Prediction images: `C:\drone_big\predicted_test_images\yolov8s`
- Evaluation outputs: `C:\drone_big\runs\evaluation\yolov8s`
- Precision: `0.9509`
- Recall: `0.9854`
- mAP50: `0.9906`
- mAP50-95: `0.8365`
- Inference speed: `29.1845` ms/image

### yolov8m

- Weights: `C:\drone_big\runs\train\pineapple_yolov8m\weights\best.pt`
- Prediction images: `C:\drone_big\predicted_test_images\yolov8m`
- Evaluation outputs: `C:\drone_big\runs\evaluation\yolov8m`
- Precision: `0.9546`
- Recall: `0.9763`
- mAP50: `0.9897`
- mAP50-95: `0.8475`
- Inference speed: `27.1604` ms/image

### yolov8l

- Weights: `C:\drone_big\runs\train\pineapple_yolov8l\weights\best.pt`
- Prediction images: `C:\drone_big\predicted_test_images\yolov8l`
- Evaluation outputs: `C:\drone_big\runs\evaluation\yolov8l`
- Precision: `0.9609`
- Recall: `0.9682`
- mAP50: `0.9903`
- mAP50-95: `0.8333`
- Inference speed: `25.6052` ms/image

### yolov8x

- Weights: `C:\drone_big\runs\train\pineapple_yolov8x\weights\best.pt`
- Prediction images: `C:\drone_big\predicted_test_images\yolov8x`
- Evaluation outputs: `C:\drone_big\runs\evaluation\yolov8x`
- Precision: `0.9540`
- Recall: `0.9764`
- mAP50: `0.9901`
- mAP50-95: `0.8370`
- Inference speed: `42.2261` ms/image
