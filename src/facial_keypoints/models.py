from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNNBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 30),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(inputs)
        return self.fc_layers(features)


class MobileNetFacialKeypoints(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v2(weights=weights)

        original_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                self.model.features[0][0].weight.copy_(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 30),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def build_model(model_name: str, pretrained: bool = False) -> nn.Module:
    normalized = model_name.lower().replace("-", "_")
    if normalized in {"baseline", "baseline_cnn", "simple_cnn"}:
        return SimpleCNNBaseline()
    if normalized in {"mobilenet", "mobilenet_v2", "mobilenet_pretrained"}:
        return MobileNetFacialKeypoints(pretrained=pretrained)
    raise ValueError(f"Unsupported model_name: {model_name}")
