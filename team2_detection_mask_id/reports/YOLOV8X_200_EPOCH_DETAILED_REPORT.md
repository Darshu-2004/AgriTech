# yolov8x_200_epochs_detailed_report

## Training Setup

- Base model: `yolov8x.pt`
- Epochs: `200`
- Batch size: `2`
- Image size: `640`
- Device: `'0'`
- Dataset: `C:\drone_big\dataset\data.yaml`
- Optimizer mode: `auto`
- Training folder: `C:\drone_big\runs\train\pineapple_yolov8x2`
- Evaluation folder: `C:\drone_big\runs\evaluation\yolov8x_200e`
- Prediction image folder: `C:\drone_big\predicted_test_images\yolov8x_200e`
- Best weights: `C:\drone_big\runs\train\pineapple_yolov8x2\weights\best.pt`

## Training Summary

- Recorded epochs in `results.csv`: `200`
- Best validation epoch by mAP50-95: `188`
- Time at best epoch: `2418.0100 s (00:40:18)`
- Total recorded training time: `2588.2100 s (00:43:08)`

## Best Epoch Metrics

- Train box loss: `0.7034 `
- Train cls loss: `0.3314 `
- Train dfl loss: `0.8379 `
- Val precision: `0.9656 `
- Val recall: `0.9832 `
- Val mAP50: `0.9916 `
- Val mAP50-95: `0.8710 `
- Val box loss: `0.6067 `
- Val cls loss: `0.2956 `
- Val dfl loss: `0.8265 `

## Final Epoch Metrics

- Final epoch: `200`
- Final train box loss: `0.6437 `
- Final train cls loss: `0.3038 `
- Final train dfl loss: `0.8327 `
- Final val precision: `0.9670 `
- Final val recall: `0.9832 `
- Final val mAP50: `0.9915 `
- Final val mAP50-95: `0.8683 `

## Test Set Evaluation

- Precision: `0.9593`
- Recall: `0.9764`
- mAP50: `0.9910`
- mAP50-95: `0.8509`
- Fitness: `0.8509`
- Preprocess speed: `1.5674` ms/image
- Inference speed: `40.7350` ms/image
- Postprocess speed: `1.3398` ms/image
- Predicted test images saved: `20`

## Artifacts

- Training curves: `C:\drone_big\runs\train\pineapple_yolov8x2\results.png`
- Training confusion matrix: `C:\drone_big\runs\train\pineapple_yolov8x2\confusion_matrix.png`
- Evaluation confusion matrix: `C:\drone_big\runs\evaluation\yolov8x_200e\confusion_matrix.png`
- Precision-recall curve: `C:\drone_big\runs\evaluation\yolov8x_200e\PR_curve.png`
- Precision curve: `C:\drone_big\runs\evaluation\yolov8x_200e\P_curve.png`
- Recall curve: `C:\drone_big\runs\evaluation\yolov8x_200e\R_curve.png`
- F1 curve: `C:\drone_big\runs\evaluation\yolov8x_200e\F1_curve.png`
