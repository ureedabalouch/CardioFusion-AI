# ConvNeXt-Tiny Model

This directory documents the image-based model used in CardioFusion AI.

## Model Details

- **Architecture:** ConvNeXt-Tiny
- **Task:** Binary classification of ECG waveform images
- **Classes:** Normal and Abnormal
- **Checkpoint:** `convnext_binary_best_fulltraining.pth`

## Download the Checkpoint

The trained checkpoint is hosted on Hugging Face because it exceeds GitHub's standard individual-file upload limit.

**Hugging Face Repository:**  
https://huggingface.co/ureedabalouch/cardiofusion-convnext-tiny

## Usage

Load the checkpoint using the model architecture and preprocessing expected by the CardioFusion AI source code. Refer to the project repository for implementation details.

## Disclaimer

This model is part of an academic research project. Its predictions are not a medical diagnosis and must not be used as the sole basis for medical decisions.
