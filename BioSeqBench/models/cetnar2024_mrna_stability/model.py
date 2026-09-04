"""Compact non-pretrained PyTorch surrogate for Cetnar 2024."""

import torch
from torch import nn


class CetnarStabilityCNNBiLSTM(nn.Module):
    """Motif CNN followed by a bidirectional LSTM and masked mean pooling.

    Cetnar et al. fitted a LightGBM model to engineered sequence/structure
    features rather than publishing a neural network.  This is the benchmark's
    sequence-only, non-pretrained PyTorch surrogate; it intentionally does not
    claim to reproduce the authors' LightGBM implementation.
    """

    def __init__(self, input_channels: int = 4, output_dim: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 96, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 96, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(96, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, mask: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(sequence_tensor).transpose(1, 2)
        encoded, _ = self.rnn(encoded)
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.head(pooled)
