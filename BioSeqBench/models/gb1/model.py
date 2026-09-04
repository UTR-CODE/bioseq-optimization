import torch
import torch.nn.functional as F
from torch import nn


class GB1FlipCNN(nn.Module):
    """FLIP supervised CNN: width-5/1024 conv, 2048 projection, max pool."""
    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(20, 1024, 5, padding=2)
        self.project = nn.Linear(1024, 2048)
        self.output = nn.Linear(2048, output_dim)

    def forward(self, sequence_tensor: torch.Tensor, mask: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(sequence_tensor.clamp_min(1) - 1, num_classes=20).float()
        x = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        x = F.relu(self.project(x))
        x = x.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(1)
        return self.output(x)

