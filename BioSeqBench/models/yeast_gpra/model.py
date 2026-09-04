"""CNN surrogate for the Vaishnav et al. yeast GPRA promoter landscapes."""

import torch
from torch import nn


class YeastGPRACNN(nn.Module):
    def __init__(self, output_dim: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 128, 13, padding=6), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence_tensor))
