# Full YOLOv8 Training Report

## Dataset Preparation

- Source images exported: `242`
- Labeled image/label pairs kept: `194`
- Unlabeled images removed: `48`
- Train split: `155`
- Validation split: `19`
- Test split: `20`
- Dataset config: `C:\drone_big\dataset\data.yaml`

## Environment

- Python: `3.14.3`
- Ultralytics: `8.4.31`
- PyTorch: `2.11.0+cu128`
- GPU: `NVIDIA GeForce RTX 4060`
- CUDA detected by PyTorch: `12.8`

## Overall Summary

- Best test `mAP50-95`: `yolov8m` with `0.8475`
- Best test precision: `yolov8l` with `0.9609`
- Best test recall: `yolov8s` with `0.9854`
- Fastest saved-prediction inference: `yolov8l` with `25.6052` ms/image

## Model Comparison Table

| Model | Epochs | Batch | Device | Best Epoch | Best Val mAP50-95 | Final Val mAP50-95 | Test Precision | Test Recall | Test mAP50 | Test mAP50-95 | Test Inference ms/img | Best Weights |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| yolov8s | 100 | 16 | '0' | 94 | 0.8590 | 0.8569 | 0.9509 | 0.9854 | 0.9906 | 0.8365 | 29.1845 | `C:\drone_big\runs\detect\runs\train\pineapple_yolov8s2\weights\best.pt` |
| yolov8m | 100 | 8 | '0' | 86 | 0.8665 | 0.8656 | 0.9546 | 0.9763 | 0.9897 | 0.8475 | 27.1604 | `C:\drone_big\runs\train\pineapple_yolov8m\weights\best.pt` |
| yolov8l | 100 | 4 | '0' | 90 | 0.8567 | 0.8552 | 0.9609 | 0.9682 | 0.9903 | 0.8333 | 25.6052 | `C:\drone_big\runs\train\pineapple_yolov8l\weights\best.pt` |
| yolov8x | 100 | 2 | '0' | 100 | 0.8546 | 0.8546 | 0.9540 | 0.9764 | 0.9901 | 0.8370 | 42.2261 | `C:\drone_big\runs\train\pineapple_yolov8x\weights\best.pt` |

## Per-Model Detail

### yolov8s

**Training Setup**

- Base pretrained weights: `yolov8s.pt`
- Epoch target: `100`
- Image size: `640`
- Batch size: `16`
- Device used: `'0'`
- Optimizer mode: `auto`
- Dataset file: `C:\drone_big\dataset\data.yaml`
- Training folder: `C:\drone_big\runs\detect\runs\train\pineapple_yolov8s2`
- Evaluation folder: `C:\drone_big\runs\evaluation\yolov8s`
- Prediction image folder: `C:\drone_big\predicted_test_images\yolov8s`
- Best weights: `C:\drone_big\runs\detect\runs\train\pineapple_yolov8s2\weights\best.pt`
- Training curve image: `C:\drone_big\runs\detect\runs\train\pineapple_yolov8s2\results.png`
- Training confusion matrix: `C:\drone_big\runs\detect\runs\train\pineapple_yolov8s2\confusion_matrix.png`
- Evaluation confusion matrix: `C:\drone_big\runs\evaluation\yolov8s\confusion_matrix.png`

**Best Validation Epoch**

- Best epoch by validation `mAP50-95`: `94`
- Cumulative training time at best epoch: `816.9880 s (00:13:37)`
- Train box loss at best epoch: `0.7318 `
- Train cls loss at best epoch: `0.3578 `
- Train dfl loss at best epoch: `0.8448 `
- Validation precision at best epoch: `0.9674 `
- Validation recall at best epoch: `0.9792 `
- Validation mAP50 at best epoch: `0.9921 `
- Validation mAP50-95 at best epoch: `0.8590 `
- Validation box loss at best epoch: `0.6046 `
- Validation cls loss at best epoch: `0.3242 `
- Validation dfl loss at best epoch: `0.8109 `

**Final Training Epoch**

- Final epoch: `100`
- Total recorded training time: `840.0940 s (00:14:00)`
- Final train box loss: `0.7187 `
- Final train cls loss: `0.3497 `
- Final train dfl loss: `0.8431 `
- Final validation precision: `0.9665 `
- Final validation recall: `0.9815 `
- Final validation mAP50: `0.9924 `
- Final validation mAP50-95: `0.8569 `
- Final validation box loss: `0.6067 `
- Final validation cls loss: `0.3129 `
- Final validation dfl loss: `0.8107 `

**Test Set Evaluation**

- Test precision: `0.9509`
- Test recall: `0.9854`
- Test mAP50: `0.9906`
- Test mAP50-95: `0.8365`
- Test fitness: `0.8365`
- Test preprocess speed: `4.0795` ms/image
- Test inference speed: `29.1845` ms/image
- Test postprocess speed: `3.1215` ms/image
- Boxed prediction images saved: `20` files

**Artifacts and Analysis Notes**

- This is the final GPU-trained `yolov8s` run used for comparison.
- An earlier CPU run also exists at `runs\detect\runs\train\pineapple_yolov8s` and is noted separately below.

### yolov8m

**Training Setup**

- Base pretrained weights: `yolov8m.pt`
- Epoch target: `100`
- Image size: `640`
- Batch size: `8`
- Device used: `'0'`
- Optimizer mode: `auto`
- Dataset file: `C:\drone_big\dataset\data.yaml`
- Training folder: `C:\drone_big\runs\train\pineapple_yolov8m`
- Evaluation folder: `C:\drone_big\runs\evaluation\yolov8m`
- Prediction image folder: `C:\drone_big\predicted_test_images\yolov8m`
- Best weights: `C:\drone_big\runs\train\pineapple_yolov8m\weights\best.pt`
- Training curve image: `C:\drone_big\runs\train\pineapple_yolov8m\results.png`
- Training confusion matrix: `C:\drone_big\runs\train\pineapple_yolov8m\confusion_matrix.png`
- Evaluation confusion matrix: `C:\drone_big\runs\evaluation\yolov8m\confusion_matrix.png`

**Best Validation Epoch**

- Best epoch by validation `mAP50-95`: `86`
- Cumulative training time at best epoch: `598.0510 s (00:09:58)`
- Train box loss at best epoch: `0.7391 `
- Train cls loss at best epoch: `0.3540 `
- Train dfl loss at best epoch: `0.8401 `
- Validation precision at best epoch: `0.9650 `
- Validation recall at best epoch: `0.9808 `
- Validation mAP50 at best epoch: `0.9925 `
- Validation mAP50-95 at best epoch: `0.8665 `
- Validation box loss at best epoch: `0.6570 `
- Validation cls loss at best epoch: `0.3380 `
- Validation dfl loss at best epoch: `0.8345 `

**Final Training Epoch**

- Final epoch: `100`
- Total recorded training time: `732.1730 s (00:12:12)`
- Final train box loss: `0.6845 `
- Final train cls loss: `0.3342 `
- Final train dfl loss: `0.8308 `
- Final validation precision: `0.9617 `
- Final validation recall: `0.9863 `
- Final validation mAP50: `0.9923 `
- Final validation mAP50-95: `0.8656 `
- Final validation box loss: `0.6420 `
- Final validation cls loss: `0.3276 `
- Final validation dfl loss: `0.8313 `

**Test Set Evaluation**

- Test precision: `0.9546`
- Test recall: `0.9763`
- Test mAP50: `0.9897`
- Test mAP50-95: `0.8475`
- Test fitness: `0.8475`
- Test preprocess speed: `2.6516` ms/image
- Test inference speed: `27.1604` ms/image
- Test postprocess speed: `3.0985` ms/image
- Boxed prediction images saved: `20` files

**Artifacts and Analysis Notes**

- No extra run notes for this model.

### yolov8l

**Training Setup**

- Base pretrained weights: `yolov8l.pt`
- Epoch target: `100`
- Image size: `640`
- Batch size: `4`
- Device used: `'0'`
- Optimizer mode: `auto`
- Dataset file: `C:\drone_big\dataset\data.yaml`
- Training folder: `C:\drone_big\runs\train\pineapple_yolov8l`
- Evaluation folder: `C:\drone_big\runs\evaluation\yolov8l`
- Prediction image folder: `C:\drone_big\predicted_test_images\yolov8l`
- Best weights: `C:\drone_big\runs\train\pineapple_yolov8l\weights\best.pt`
- Training curve image: `C:\drone_big\runs\train\pineapple_yolov8l\results.png`
- Training confusion matrix: `C:\drone_big\runs\train\pineapple_yolov8l\confusion_matrix.png`
- Evaluation confusion matrix: `C:\drone_big\runs\evaluation\yolov8l\confusion_matrix.png`

**Best Validation Epoch**

- Best epoch by validation `mAP50-95`: `90`
- Cumulative training time at best epoch: `925.4880 s (00:15:25)`
- Train box loss at best epoch: `0.7765 `
- Train cls loss at best epoch: `0.3852 `
- Train dfl loss at best epoch: `0.8477 `
- Validation precision at best epoch: `0.9644 `
- Validation recall at best epoch: `0.9756 `
- Validation mAP50 at best epoch: `0.9918 `
- Validation mAP50-95 at best epoch: `0.8567 `
- Validation box loss at best epoch: `0.6514 `
- Validation cls loss at best epoch: `0.3403 `
- Validation dfl loss at best epoch: `0.8326 `

**Final Training Epoch**

- Final epoch: `100`
- Total recorded training time: `1073.3300 s (00:17:53)`
- Final train box loss: `0.7665 `
- Final train cls loss: `0.3711 `
- Final train dfl loss: `0.8641 `
- Final validation precision: `0.9644 `
- Final validation recall: `0.9761 `
- Final validation mAP50: `0.9920 `
- Final validation mAP50-95: `0.8552 `
- Final validation box loss: `0.6428 `
- Final validation cls loss: `0.3343 `
- Final validation dfl loss: `0.8299 `

**Test Set Evaluation**

- Test precision: `0.9609`
- Test recall: `0.9682`
- Test mAP50: `0.9903`
- Test mAP50-95: `0.8333`
- Test fitness: `0.8333`
- Test preprocess speed: `2.7291` ms/image
- Test inference speed: `25.6052` ms/image
- Test postprocess speed: `2.3579` ms/image
- Boxed prediction images saved: `20` files

**Artifacts and Analysis Notes**

- No extra run notes for this model.

### yolov8x

**Training Setup**

- Base pretrained weights: `yolov8x.pt`
- Epoch target: `100`
- Image size: `640`
- Batch size: `2`
- Device used: `'0'`
- Optimizer mode: `auto`
- Dataset file: `C:\drone_big\dataset\data.yaml`
- Training folder: `C:\drone_big\runs\train\pineapple_yolov8x`
- Evaluation folder: `C:\drone_big\runs\evaluation\yolov8x`
- Prediction image folder: `C:\drone_big\predicted_test_images\yolov8x`
- Best weights: `C:\drone_big\runs\train\pineapple_yolov8x\weights\best.pt`
- Training curve image: `C:\drone_big\runs\train\pineapple_yolov8x\results.png`
- Training confusion matrix: `C:\drone_big\runs\train\pineapple_yolov8x\confusion_matrix.png`
- Evaluation confusion matrix: `C:\drone_big\runs\evaluation\yolov8x\confusion_matrix.png`

**Best Validation Epoch**

- Best epoch by validation `mAP50-95`: `100`
- Cumulative training time at best epoch: `1686.1200 s (00:28:06)`
- Train box loss at best epoch: `0.7801 `
- Train cls loss at best epoch: `0.3787 `
- Train dfl loss at best epoch: `0.8665 `
- Validation precision at best epoch: `0.9633 `
- Validation recall at best epoch: `0.9767 `
- Validation mAP50 at best epoch: `0.9922 `
- Validation mAP50-95 at best epoch: `0.8546 `
- Validation box loss at best epoch: `0.6431 `
- Validation cls loss at best epoch: `0.3286 `
- Validation dfl loss at best epoch: `0.8333 `

**Final Training Epoch**

- Final epoch: `100`
- Total recorded training time: `1686.1200 s (00:28:06)`
- Final train box loss: `0.7801 `
- Final train cls loss: `0.3787 `
- Final train dfl loss: `0.8665 `
- Final validation precision: `0.9633 `
- Final validation recall: `0.9767 `
- Final validation mAP50: `0.9922 `
- Final validation mAP50-95: `0.8546 `
- Final validation box loss: `0.6431 `
- Final validation cls loss: `0.3286 `
- Final validation dfl loss: `0.8333 `

**Test Set Evaluation**

- Test precision: `0.9540`
- Test recall: `0.9764`
- Test mAP50: `0.9901`
- Test mAP50-95: `0.8370`
- Test fitness: `0.8370`
- Test preprocess speed: `4.3208` ms/image
- Test inference speed: `42.2261` ms/image
- Test postprocess speed: `3.7277` ms/image
- Boxed prediction images saved: `20` files

**Artifacts and Analysis Notes**

- No extra run notes for this model.

## Additional Notes

- Short comparison report: `C:\drone_big\MODEL_COMPARISON.md`
- The `yolov8s` family has two run folders because an earlier CPU attempt existed before the final GPU-standardized run.
- Final comparison and prediction outputs were produced from the GPU-trained runs.
