"""Non-pretrained CNN baselines for synthetic E. coli promoter libraries."""

import torch
from torch import nn


class EcoliPromoterCNN(nn.Module):
    """CNN strength predictor aligned with the Zhou et al. L3 predictor class."""

    def __init__(self, output_dim: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 128, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(sequence_tensor))
