from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.model_architectures import SignalCNN, create_convnext


PROJECT_DIR = Path(__file__).resolve().parents[1]

SIGNAL_CHECKPOINT = (
    PROJECT_DIR / "models" / "signal_cnn" / "best_model.pth"
)
IMAGE_CHECKPOINT = (
    PROJECT_DIR / "experiments" / "ecg_convnext" /
    "checkpoints" / "convnext_binary_best_fulltraining.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


# Signal CNN: preserve its existing checkpoint and preprocessing.
signal_checkpoint = load_checkpoint(SIGNAL_CHECKPOINT)
signal_model = SignalCNN()
signal_model.load_state_dict(signal_checkpoint["model_state"])
signal_model.to(DEVICE).eval()
SIGNAL_CLASSES = signal_checkpoint["class_mapping"]


# ConvNeXt: load the exact architecture and checkpoint used in training.
image_checkpoint = load_checkpoint(IMAGE_CHECKPOINT)
image_model = create_convnext()
image_model.load_state_dict(
    image_checkpoint["model_state_dict"],
    strict=True,
)
image_model.to(DEVICE).eval()
IMAGE_CLASSES = image_checkpoint["class_to_idx"]


# Match the preprocessing recorded in the ConvNeXt checkpoint.
image_transform = transforms.Compose([
    transforms.Resize(
        256,
        interpolation=InterpolationMode.BICUBIC,
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def prepare_signal(signal):
    """Prepare one already-digitized, 12-lead ECG for Signal CNN."""
    signal = np.asarray(signal, dtype=np.float32)

    if signal.shape != (1000, 12):
        raise ValueError(
            "Expected a signal shaped (1000, 12): "
            "1000 samples and 12 leads."
        )

    if not np.isfinite(signal).all():
        raise ValueError("The signal contains NaN or infinite values.")

    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    signal = (signal - mean) / np.maximum(std, 1e-6)

    tensor = torch.from_numpy(signal.T.copy()).unsqueeze(0)
    return tensor.to(DEVICE)


def predict_signal(signal):
    """Return the Signal CNN's predicted class and model scores."""
    tensor = prepare_signal(signal)

    with torch.inference_mode():
        logits = signal_model(tensor)
        probabilities = F.softmax(
            logits.float(), dim=1
        )[0].cpu().tolist()

    predicted_index = int(np.argmax(probabilities))
    predicted_class = next(
        name for name, index in SIGNAL_CLASSES.items()
        if index == predicted_index
    )

    return {
        "model": "Signal CNN",
        "predicted_class": predicted_class,
        "scores": {
            name: float(probabilities[index])
            for name, index in SIGNAL_CLASSES.items()
        },
    }


def predict_image(image):
    """Return ConvNeXt's predicted class and model scores."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")
    tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = image_model(tensor)
        probabilities = F.softmax(
            logits.float(), dim=1
        )[0].cpu().tolist()

    predicted_index = int(np.argmax(probabilities))
    predicted_class = next(
        name for name, index in IMAGE_CLASSES.items()
        if index == predicted_index
    )

    return {
        "model": "ConvNeXt-Tiny",
        "predicted_class": predicted_class,
        "scores": {
            name: float(probabilities[index])
            for name, index in IMAGE_CLASSES.items()
        },
    }
