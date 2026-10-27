# GreenPod Labs — Assignment 2 Report
## Image Classification Using the Provided Orange Dataset

**Candidate:** [Your name]  
**Date:** [Date]  
**Dataset link:** https://drive.google.com/file/d/1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET/view

> **Dataset discrepancy:** The supplied Drive ZIP contains orange image classes (`blackspot`, `canker`, `fresh`, `grenning`), not avocado ripening stages. This report is structured for the actual supplied dataset. Confirm with GreenPod Labs whether the correct dataset was shared before submission.

## 1. Abstract
This project compares two image classification pipelines on the supplied four-class orange image dataset: (1) engineered HSV/CIELAB color statistics and LBP texture descriptors classified by an SVM, and (2) MobileNetV2 transfer learning. Both are evaluated on the same held-out test folder using accuracy, macro precision, macro recall, macro F1-score and confusion matrices.

## 2. Dataset understanding
Dataset classes: `blackspot`, `canker`, `fresh`, `grenning`.

Provided image counts:
| Class | Train | Test |
|---|---:|---:|
| blackspot | 184 | 22 |
| canker | 179 | 22 |
| fresh | 281 | 33 |
| grenning | 347 | 22 |
| **Total** | **991** | **99** |

Training class imbalance: `grenning` has the most training examples (347) and `canker` the fewest (179), a max/min ratio of approximately 1.94. The supplied test set has 22 images for three classes and 33 for `fresh`. Counts are based on the uploaded ZIP; confirm after running EDA.

Discuss:
- Image size/resolution range from `outputs/eda/image_size_summary.csv`.
- Corrupted/unreadable images from `outputs/eda/unreadable_images.csv`.
- Visible differences in lighting, shadows, backgrounds, camera angles, focus and fruit appearance from `outputs/eda/sample_contact_sheet.png`.
- Possible label noise, duplicate/near-duplicate images, and dataset collection bias. Only report checks actually performed.

## 3. Preprocessing and split
The supplied `train/` and `test/` split is preserved. A stratified validation subset (15% of the supplied training data) is created for model selection. The test set is not used for hyperparameter tuning. The classical model resizes images to 128×128; MobileNetV2 resizes to 224×224 and applies ImageNet normalization. Training augmentation uses horizontal flips, small rotations and mild color jitter.

## 4. Approach 1 — Classical computer vision + SVM
HSV and CIELAB channel means, standard deviations and percentiles summarize color. Local Binary Pattern (LBP) histograms describe local texture. A standardized feature vector is classified by an RBF-kernel SVM with balanced class weights.

**Reason for selection:** Fast and interpretable baseline; tests whether engineered color and texture features are sufficient. Limitations include sensitivity to lighting/background changes and limited ability to learn complex patterns.

## 5. Approach 2 — MobileNetV2 transfer learning
An ImageNet-pretrained MobileNetV2 is used with a new output layer for the four dataset classes. The classification head is trained with cross-entropy and AdamW. Validation loss chooses the best checkpoint and controls early stopping.

**Reason for selection:** A lightweight CNN can learn richer visual patterns than manually engineered features and is a reasonable candidate for constrained inference. It needs more compute and may overfit a small or biased dataset.

## 6. Metrics and comparison
- **Accuracy:** overall proportion of correct test predictions.
- **Macro precision:** class-balanced measure of how reliable predictions are.
- **Macro recall:** class-balanced measure of how many examples are recovered.
- **Macro F1-score:** balances precision and recall with equal class weighting; primary metric under imbalance.
- **Confusion matrix:** reveals specific class confusions.
- **Latency/model size:** relevant for production deployment and should be measured on named hardware.

| Test metric | Classical SVM | MobileNetV2 |
|---|---:|---:|
| Accuracy | [actual] | [actual] |
| Macro precision | [actual] | [actual] |
| Macro recall | [actual] | [actual] |
| Macro F1-score | [actual] | [actual] |
| Per-image latency (ms) | [measured] | [measured] |
| Model size (MB) | [measured] | [measured] |

Insert the generated confusion matrix images and discuss the main error patterns. Choose based on macro F1, minority-class recall, confusion patterns, latency and model size, not accuracy alone.

## 7. Example run
Command:
```powershell
python src/05_demo.py --image "data/split/test/fresh/REPLACE_WITH_REAL_FILENAME.jpg"
```
Replace the filename with an existing test image. Paste the actual terminal output and include the image and prediction screenshot. Explain whether both models agree and whether the prediction is correct according to the dataset label.

## 8. Mandatory questions

### 8.1 In production, how does your model perform?
[Insert actual test metrics and measured single-image latency, hardware, and timing method. Discuss lighting, backgrounds, fruit appearance, camera and variety shifts. Test-set performance alone does not establish production readiness.]

### 8.2 What are the drawbacks of your model?
Visual appearance may overlap across classes. Color-based features can be affected by lighting and backgrounds. A single image may not show all surface defects. Labels may be noisy, and the dataset may not represent all real-world conditions. The supplied classes are orange-related, so the model cannot be claimed to detect avocado ripeness.

### 8.3 If given more time, what would you improve?
Collect more balanced images from multiple sources and lighting conditions, verify labels and duplicates, perform external validation, tune the models, calibrate confidence, test quantization/ONNX and measure performance on deployment hardware.

### 8.4 Where did you spend the most time while solving your assignment?
[Describe your actual work on dataset inspection, feature engineering, model training, debugging and error analysis. Add percentages only if you tracked your time.]

## 9. Conclusion
[Summarize actual results, which approach performed better, per-class weaknesses and the speed/accuracy trade-off. State the dataset mismatch clearly until GreenPod Labs confirms the intended dataset.]
