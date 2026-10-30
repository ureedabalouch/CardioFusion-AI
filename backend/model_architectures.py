import torch
import torch.nn as nn
from torchvision import models


class SignalCNN(nn.Module):
    """12-lead ECG signal classifier."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_densenet121():
    """DenseNet-121 trained from scratch with a two-class head."""
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model
def create_convnext():
    """Create the ConvNeXt-Tiny architecture used during training."""
    import timm

    return timm.create_model(
        "convnext_tiny.fb_in22k",
        pretrained=False,
        num_classes=2,
    )
