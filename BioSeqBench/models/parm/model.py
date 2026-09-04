"""PARM-style convolutional promoter-activity predictor."""

import torch
from torch import nn


class PARMCNN(nn.Module):
    """Lightweight, non-pretrained CNN for the PARM Log2RPM targets."""

    def __init__(self, output_dim: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 128, 15, padding=7), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 128, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence_tensor))
