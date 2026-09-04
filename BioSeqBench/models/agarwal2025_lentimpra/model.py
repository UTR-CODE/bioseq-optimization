"""Non-pretrained MPRAnn/LegNet-style convolutional lentiMPRA baseline."""

import torch
from torch import nn


class AgarwalMPRACNN(nn.Module):
    def __init__(self, output_dim: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 256, 15, padding=7), nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 128, 7, padding=3), nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence_tensor))
