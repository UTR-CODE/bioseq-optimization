from __future__ import annotations

import torch
from torch import nn


class RNAConvBiLSTM(nn.Module):
    def __init__(self, input_channels: int = 4, output_dim: int = 1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(input_channels, 64, 5, padding=2), nn.ReLU())
        self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, mask: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        x = self.conv(sequence_tensor).transpose(1, 2)
        x, _ = self.lstm(x)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)

