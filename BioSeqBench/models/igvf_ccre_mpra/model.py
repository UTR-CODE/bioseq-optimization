import torch
from torch import nn


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, hidden, 1), nn.SiLU(), nn.Conv1d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class EfficientBlock1d(nn.Module):
    def __init__(self, channels: int, kernel: int = 7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.BatchNorm1d(channels * 2), nn.SiLU(),
            nn.Conv1d(channels * 2, channels * 2, kernel, padding=kernel // 2, groups=channels * 2),
            nn.BatchNorm1d(channels * 2), nn.SiLU(), SqueezeExcite1d(channels * 2),
            nn.Conv1d(channels * 2, channels, 1), nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x + self.block(x))


class IGVFLegNet(nn.Module):
    """MPRALegNet-style EfficientNetV2/SE backbone with cell-context embedding."""
    def __init__(self, num_cell_types: int, output_dim: int = 1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, 128, 15, padding=7), nn.BatchNorm1d(128), nn.SiLU())
        self.blocks = nn.Sequential(EfficientBlock1d(128), nn.MaxPool1d(2), EfficientBlock1d(128), nn.MaxPool1d(2), EfficientBlock1d(128))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cell_embedding = nn.Embedding(num_cell_types, 16)
        self.head = nn.Sequential(nn.Linear(144, 64), nn.SiLU(), nn.Dropout(0.2), nn.Linear(64, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, cell_type_id: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.blocks(self.stem(sequence_tensor))).squeeze(-1)
        return self.head(torch.cat([x, self.cell_embedding(cell_type_id)], dim=1))

