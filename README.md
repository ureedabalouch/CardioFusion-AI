# CardioFusion AI 🫀

**Deep Learning-Based ECG Classification Using Signal CNN and ConvNeXt-Tiny**

CardioFusion AI is a deep learning project that classifies 12-lead electrocardiogram (ECG) data into two categories: **Normal** and **Abnormal**. The project explores two approaches to ECG classification: learning from raw ECG signals and learning from ECG waveform images.

> **Disclaimer:** CardioFusion AI is an academic project and is not intended to replace professional medical advice, clinical diagnosis, or evaluation by a qualified healthcare professional.

## 📌 Project Overview

Electrocardiograms (ECGs) are used to record the electrical activity of the heart. Deep learning can help identify patterns in ECG data and support research into automated ECG classification.

CardioFusion AI uses two separate deep learning models:

1. **Signal CNN** — processes 12-lead ECG signal data.
2. **ConvNeXt-Tiny** — processes images of ECG waveforms.

Both models target binary classification: Normal vs. Abnormal. Their predictions and evaluation results will be assessed separately.

## 🎯 Objectives

- Develop a deep learning model for 12-lead ECG signal classification.
- Train an image-based model using ECG waveform images.
- Compare signal-based and image-based classification approaches.
- Evaluate model performance using appropriate classification metrics.
- Develop a web interface to demonstrate the project.
- Document the dataset, training process, model architecture, and evaluation.

## 🧠 Models

### 1. Signal CNN

The Signal CNN learns patterns directly from numerical ECG signals.

- **Input:** 12-lead ECG signal.
- **Expected signal shape:** 1000 time steps × 12 leads before model preprocessing.
- **Model input format:** 12 leads × 1000 time steps.
- **Task:** Binary classification.
- **Output classes:** Normal and Abnormal.

### 2. ConvNeXt-Tiny

ConvNeXt-Tiny is an image-based deep learning architecture used to learn visual patterns from ECG waveform images.

- **Input:** ECG waveform image.
- **Image preprocessing:** Resize, center crop, and ImageNet normalization.
- **Model:** ConvNeXt-Tiny.
- **Task:** Binary classification.
- **Output classes:** Normal and Abnormal.

The two models use different input modalities and should be evaluated independently before comparing their performance.

## 📊 Dataset

This project uses the **PTB-XL ECG dataset**, a publicly available collection of electrocardiogram records.

- **Dataset name:** PTB-XL
- **Source:** PhysioNet
- **Records:** 21,799 ECG records in the original dataset.
- **Selected task:** Normal vs. Abnormal classification.
- **Signal data:** 12-lead ECG recordings.
- **Image data:** ECG waveform images generated for image-based model training.

**Official dataset:** https://physionet.org/content/ptb-xl/

The project uses a processed manifest to associate ECG records with binary labels and dataset splits. The exact inclusion and exclusion criteria should be documented alongside the final dataset manifest.

### Dataset handling

The original PTB-XL data should be obtained from its official source and used according to its license and terms. This repository is intended to contain project code, documentation, and model artifacts rather than an unnecessary duplicate of the full raw dataset.

## ⚙️ Technology Stack

The project may use the following technologies, depending on the final repository contents:

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Pandas**
- **scikit-learn**
- **WFDB**
- **Pillow**
- **React / TypeScript** for the web interface, if included

The final dependency files should reflect the packages actually required by the published code.

## 🔄 Project Workflow

1. Obtain the PTB-XL dataset.
2. Read ECG records and their metadata.
3. Prepare binary Normal and Abnormal labels.
4. Create training, validation, and test splits.
5. Preprocess ECG signals for Signal CNN.
6. Generate and preprocess ECG waveform images for ConvNeXt-Tiny.
7. Train the two models separately.
8. Evaluate each model on the appropriate held-out test set.
9. Save the trained model weights.
10. Integrate the models into the project interface or prediction API.
11. Document results and limitations.

## 📈 Evaluation

Model performance should be evaluated on a held-out test set. The following metrics are suitable for the binary classification task:

- Accuracy
- Precision
- Recall / sensitivity
- Specificity
- Macro F1-score
- ROC-AUC
- Confusion matrix

Final metric values should be added only after evaluating the correct saved checkpoints. Results from one architecture must not be attributed to another.

## 📁 Repository Structure

The repository is intended to follow this structure as the project files are added:

```text
CardioFusion-AI/
├── README.md
├── backend/
│   ├── models/
│   └── prediction_service.py
├── models/
│   ├── signal_cnn/
│   │   └── best_model.pth
│   └── convnext_tiny/
│       └── best_model.pth
├── data/
│   └── README.md
├── notebooks/
├── results/
├── docs/
├── requirements.txt
└── .gitignore
```

*The structure is a planned layout; files should be added only when they exist and are ready to publish.*

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/CardioFusion-AI.git
cd CardioFusion-AI
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Set up Python

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

After the project’s `requirements.txt` has been prepared:

```bash
pip install -r requirements.txt
```

### 4. Obtain the dataset

Download PTB-XL from the official PhysioNet page and follow the project's documented data preparation instructions.

### 5. Run the project

Detailed training and inference commands will be documented after the final scripts, checkpoints, and dependencies have been added to the repository.

## ⚠️ Limitations

- The project performs binary Normal vs. Abnormal classification; it does not claim to diagnose specific diseases.
- Predictions depend on the quality, preprocessing, and distribution of the input data.
- ECG waveform images must be suitable for the image model; arbitrary images should not be treated as valid ECG inputs.
- Test results do not automatically establish clinical reliability or generalization to every patient population.
- The project is intended for educational and research purposes.

## 🔮 Future Improvements

- Complete and document reproducible training pipelines.
- Compare Signal CNN and ConvNeXt-Tiny using consistent evaluation procedures.
- Add confusion matrices and metric visualizations.
- Improve input validation and error handling.
- Provide a reproducible inference interface.
- Explore further validation on suitable independent datasets.

## 👩‍💻 Project Information

**Project:** CardioFusion AI  
**Domain:** Deep Learning / Healthcare AI  
**Task:** 12-lead ECG Normal vs. Abnormal classification  
**Models:** Signal CNN and ConvNeXt-Tiny  
**Dataset:** PTB-XL

## 📄 License

The repository's license should be selected before public release. The PTB-XL dataset has its own licensing and attribution requirements; consult the official dataset page before redistributing any dataset files.

---

**Disclaimer:** CardioFusion AI is an academic research project and is not a medical device. Its predictions must not be used as the sole basis for medical decisions.
