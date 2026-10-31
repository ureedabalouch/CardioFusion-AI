# CardioFusion AI 🫀

### Deep Learning-Based ECG Classification Using Signal CNN and ConvNeXt-Tiny

CardioFusion AI is an academic deep learning project that classifies 12-lead electrocardiogram (ECG) records into two categories: **Normal** and **Abnormal**. It explores two separate approaches: learning from numerical ECG signals and learning from ECG waveform images.

> **Disclaimer:** This project is intended for academic research and educational purposes only. It is not a medical device and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 📌 Project Overview

Electrocardiograms record the electrical activity of the heart. Deep learning models can learn patterns from ECG data and support research into automated ECG classification.

CardioFusion AI explores two model architectures:

- **Signal CNN:** Processes numerical 12-lead ECG signals.
- **ConvNeXt-Tiny:** Processes images of ECG waveforms.

Both models address the same binary classification task but use different input formats. They are trained and evaluated separately; this project does not claim that a multimodal fusion model has been implemented.

## 🎯 Project Objectives

- Develop a deep learning model for numerical ECG signal classification.
- Explore image-based ECG classification.
- Compare signal-based and image-based approaches.
- Evaluate model performance using classification metrics.
- Provide a backend API for ECG predictions.
- Document the dataset, model artifacts, results, and setup instructions.

## 🧠 Models

### 1. Signal CNN

The Signal CNN learns patterns directly from numerical ECG recordings.

- **Input:** 12-lead ECG signal.
- **Expected raw record shape:** 1000 time steps × 12 leads.
- **Task:** Binary classification.
- **Output classes:** Normal and Abnormal.
- **Checkpoint:** `models/signal_cnn/best_model.pth`

### 2. ConvNeXt-Tiny

ConvNeXt-Tiny is a convolutional neural network architecture used to learn visual patterns from ECG waveform images.

- **Input:** ECG waveform image.
- **Preprocessing:** Image resizing, cropping, and normalization.
- **Task:** Binary classification.
- **Output classes:** Normal and Abnormal.
- **Checkpoint hosting:** [Hugging Face — CardioFusion ConvNeXt-Tiny](https://huggingface.co/ureedabalouch/cardiofusion-convnext-tiny)

The ConvNeXt-Tiny checkpoint is hosted on Hugging Face to avoid storing another large model file in the GitHub repository. Follow the model documentation for checkpoint access and setup.

## 📊 Dataset

CardioFusion AI uses the **PTB-XL ECG dataset**, available through PhysioNet.

- **Dataset:** PTB-XL
- **Original dataset size:** 21,799 ECG records.
- **Signal format:** 12-lead ECG recordings.
- **Classification task:** Normal vs. Abnormal.
- **Use in this project:** Signal-based and image-based model development.

**Official dataset:** https://physionet.org/content/ptb-xl/

The project uses processed labels and data splits for the binary classification task. The original dataset should be obtained from its official source and handled according to its applicable licence and terms.

The full raw dataset and generated ECG image collection are not included in this repository.

## 📈 Model Evaluation

The models are evaluated using metrics appropriate for binary classification, including accuracy, Macro-F1, and area under the receiver operating characteristic curve (AUROC).

### Signal CNN

| Metric | Reported result |
|---|---:|
| Accuracy | 84.53% |
| Macro-F1 | 84.45% |
| AUROC | 92.30% |

### ConvNeXt-Tiny

| Metric | Preliminary result |
|---|---:|
| Accuracy | 78.64% |
| Macro-F1 | 77.14% |
| AUROC | 89.65% |

**Important:** The ConvNeXt-Tiny results are preliminary and require class-mapping verification before being treated as final. The results should not be considered a definitive comparison until the evaluation labels, predictions, and metric calculations have been checked consistently.

These figures represent experimental model performance, not clinical validation.

## 🔄 Project Workflow

1. Obtain the PTB-XL dataset.
2. Read ECG recordings and metadata.
3. Prepare Normal and Abnormal labels.
4. Prepare training, validation, and test splits.
5. Preprocess numerical signals for Signal CNN.
6. Generate and preprocess ECG waveform images for ConvNeXt-Tiny.
7. Train the models separately.
8. Evaluate the models using held-out test data.
9. Save and document the selected model checkpoints.
10. Integrate model inference into the backend API.
11. Review evaluation results and document limitations.

## 📁 Repository Structure

The repository is organized to keep essential application code, model artifacts, and documentation in one place without uploading the complete dataset or unnecessary training files.

```text
CardioFusion-AI/
├── backend/
│   ├── main.py
│   ├── model_architectures.py
│   ├── prediction_service.py
│   └── requirements.txt
├── data/
│   └── README.md
├── models/
│   ├── convnext_tiny/
│   │   └── README.md
│   └── signal_cnn/
│       ├── README.md
│       └── best_model.pth
├── results/
│   └── README.md
└── README.md
```

### Folder descriptions

- `backend/` — API application, prediction service, model architecture code, and Python dependencies.
- `data/` — Documentation about the dataset and data handling.
- `models/signal_cnn/` — Signal CNN checkpoint and model documentation.
- `models/convnext_tiny/` — Documentation and instructions for accessing the hosted ConvNeXt-Tiny checkpoint.
- `results/` — Documentation of evaluation results and related artifacts.
- `README.md` — Main project overview and setup guide.

## ⚙️ Technology Stack

- Python
- PyTorch
- Torchvision
- FastAPI
- NumPy
- Pandas
- scikit-learn
- WFDB
- Pillow
- Hugging Face for external model checkpoint hosting

The installed packages required by the backend are listed in `backend/requirements.txt`.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ureedabalouch/CardioFusion-AI.git
cd CardioFusion-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install backend dependencies

From the repository root, run:

```bash
pip install -r backend/requirements.txt
```

### 4. Configure model checkpoints

- Confirm that the Signal CNN checkpoint is present at `models/signal_cnn/best_model.pth`.
- Follow `models/convnext_tiny/README.md` to access the hosted ConvNeXt-Tiny checkpoint.
- Ensure the backend's configured checkpoint paths match the actual locations.

### 5. Start the API

Run the following command from the `backend/` directory, assuming the FastAPI application is exposed as `app` in `main.py`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

If the project uses additional environment variables or a different checkpoint-loading configuration, follow the instructions in the backend code and model documentation.

### 6. Explore the API

When the backend is running locally, open:

- API base URL: http://localhost:8000/
- Health endpoint: http://localhost:8000/health
- Interactive API documentation: http://localhost:8000/docs

The prediction endpoints are intended for signal-based and image-based ECG classification. Refer to the backend implementation and interactive API documentation for the accepted input formats and request schemas.

## ⚠️ Limitations

- The task is limited to binary Normal vs. Abnormal classification.
- The models do not identify or confirm specific diseases.
- ConvNeXt-Tiny evaluation results require class-mapping verification.
- Performance may depend on data quality, preprocessing, and the distribution of ECG records.
- Experimental test results do not establish clinical reliability or generalization to every patient population.
- Predictions and confidence scores should not be interpreted as calibrated clinical probabilities unless calibration has been independently established.

## 🔮 Future Improvements

- Verify and finalize the image-model evaluation.
- Improve signal and image preprocessing.
- Explore hyperparameter optimization.
- Improve prediction interface usability.
- Investigate explainable AI methods.
- Explore multiclass ECG classification and additional independent evaluation.

## 📚 References and Resources

- **GitHub repository:** https://github.com/ureedabalouch/CardioFusion-AI
- **PTB-XL dataset:** https://physionet.org/content/ptb-xl/
- **ConvNeXt-Tiny checkpoint:** https://huggingface.co/ureedabalouch/cardiofusion-convnext-tiny

## 📄 Project Information

- **Project name:** CardioFusion AI
- **Domain:** Deep Learning / Healthcare AI
- **Task:** 12-lead ECG Normal vs. Abnormal classification
- **Models:** Signal CNN and ConvNeXt-Tiny
- **Dataset:** PTB-XL

---

*CardioFusion AI — Building a foundation for AI-assisted ECG analysis.*

