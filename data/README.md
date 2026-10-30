# Dataset Documentation

## Dataset Name

PTB-XL — A Large Publicly Available Electrocardiography Dataset

## Official Source

https://physionet.org/content/ptb-xl/

## Dataset Description

CardioFusion AI uses the PTB-XL dataset for ECG classification research. PTB-XL contains 12-lead electrocardiogram (ECG) recordings and associated metadata.

The project uses the dataset to develop two binary classification models:

- **Signal CNN:** Learns from numerical ECG signals.
- **ConvNeXt-Tiny:** Learns from ECG waveform images generated for image-based classification.

## Classification Task

The project's target classes are:

- **Normal**
- **Abnormal**

These are the project's binary labels and should not be interpreted as specific disease diagnoses.

## Data Preparation

The project workflow includes:

1. Reading ECG metadata and recordings.
2. Preparing binary classification labels.
3. Creating training, validation, and test splits.
4. Preprocessing ECG signals for the Signal CNN.
5. Generating and preprocessing ECG waveform images for ConvNeXt-Tiny.

## Dataset Access

Obtain the original dataset from its official PhysioNet page:

https://physionet.org/content/ptb-xl/

Review the dataset's license, citation requirements, and access instructions before using or redistributing its contents.

## Data Availability in This Repository

The full raw PTB-XL dataset is not included in this repository. Users should obtain it from the official source and follow the project-specific preprocessing instructions.

## Important Note

CardioFusion AI is an academic project. Its binary predictions are not a substitute for professional medical interpretation or clinical diagnosis.
