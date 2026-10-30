# CardioFusion AI — Model Evaluation Results

This document reports the verified test results currently available for the CardioFusion AI models.

## 1. Signal CNN

**Task:** Binary classification of 12-lead ECG records into Normal and Abnormal classes.

### Test Configuration

- Test records: 2,107
- Selected checkpoint: `best_model.pth`
- Checkpoint epoch: 1
- Class mapping: Normal = 0, Abnormal = 1

### Evaluation Metrics

| Metric | Result |
|---|---:|
| Accuracy | 84.53% |
| Abnormal precision | 90.80% |
| Abnormal recall / sensitivity | 80.92% |
| Specificity | 89.25% |
| Macro F1-score | 84.45% |
| ROC-AUC | 92.30% |

### Confusion Matrix

Rows represent actual classes; columns represent predicted classes. Class order is Normal, Abnormal.

| Actual class | Predicted Normal | Predicted Abnormal |
|---|---:|---:|
| Normal | 814 | 98 |
| Abnormal | 228 | 967 |

The model correctly classified 814 Normal records and 967 Abnormal records in this test evaluation. The confusion matrix also shows 98 Normal records predicted as Abnormal and 228 Abnormal records predicted as Normal.

## 2. ConvNeXt-Tiny

**Task:** Binary classification of ECG waveform images into Normal and Abnormal classes.

- Checkpoint: `convnext_binary_best_fulltraining.pth`
- Hosting: [Hugging Face model repository](https://huggingface.co/ureedabalouch/cardiofusion-convnext-tiny)

ConvNeXt-Tiny evaluation metrics have not yet been added because its corresponding test evaluation results have not been verified for this report.

## Interpretation and Limitations

These results describe the evaluated test set and do not establish clinical performance or suitability for medical decision-making. Performance may vary with dataset composition, preprocessing, and input quality.

CardioFusion AI is an academic project and is not intended to replace professional medical advice or clinical diagnosis.
