from __future__ import annotations

import torch
from torch import nn


class DeepBindCNN(nn.Module):
    def __init__(self, input_channels: int = 4, output_dim: int = 1, filters: int = 64, kernel_size: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(), nn.MaxPool1d(2), nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(filters, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence_tensor))


class Optimus5Prime(nn.Module):
    """Exact sequence-only CNN from Sample et al. (2019)."""
    def __init__(self, input_channels: int = 4, output_dim: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 120, 8, padding="same"), nn.ReLU(),
            nn.Conv1d(120, 120, 8, padding="same"), nn.ReLU(),
            nn.Conv1d(120, 120, 8, padding="same"), nn.ReLU(),
            nn.Flatten(),
        )
        # The paper model is specifically defined for the 50-nt library:
        # 120 channels × 50 positions after same-padded convolutions.
        self.head = nn.Sequential(nn.Linear(120 * 50, 40), nn.ReLU(), nn.Dropout(0.2), nn.Linear(40, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence_tensor))


class DeepSTARR(nn.Module):
    """Four-block Basset-derived DeepSTARR multi-task CNN."""
    def __init__(self, input_channels: int = 4, output_dim: int = 2):
        super().__init__()
        # Filter counts and widths follow the authors' training notebook.
        channels = [input_channels, 256, 60, 60, 120]
        kernels = [7, 3, 5, 3]
        blocks = []
        for in_channels, out_channels, kernel in zip(channels[:-1], channels[1:], kernels):
            blocks.extend([
                nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_channels), nn.ReLU(), nn.MaxPool1d(2),
            ])
        # 249 bp -> 124 -> 62 -> 31 -> 15 after four pool-2 operations.
        blocks.append(nn.Flatten())
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(120 * 15, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, output_dim),
        )

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence_tensor))


class ConditionalRegulatoryCNN(nn.Module):
    """Sequence + learned cell-type embedding for IGVF multi-context regression."""
    def __init__(self, num_cell_types: int, input_channels: int = 4, output_dim: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 128, 15, padding=7), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(),
        )
        self.cell_embedding = nn.Embedding(num_cell_types, 16)
        self.head = nn.Sequential(nn.Linear(144, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, cell_type_id: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(sequence_tensor), self.cell_embedding(cell_type_id)], dim=1))
