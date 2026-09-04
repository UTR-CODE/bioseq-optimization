import torch
from torch import nn


class OpenVaccineGCNGRU(nn.Module):
    """iVaccine-Deep-style graph propagation plus bidirectional GRU."""
    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.structure_embedding = nn.Embedding(10, 8, padding_idx=0)
        self.loop_embedding = nn.Embedding(8, 8, padding_idx=0)
        self.input = nn.Linear(4 + 8 + 8, 64)
        self.graph_layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(2)])
        self.gru = nn.GRU(64, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, mask: torch.Tensor, structure_tensor: torch.Tensor, loop_tensor: torch.Tensor, pairing_matrix: torch.Tensor | None = None, **_: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            sequence_tensor.transpose(1, 2), self.structure_embedding(structure_tensor), self.loop_embedding(loop_tensor),
        ], dim=-1)
        x = torch.relu(self.input(x))
        if pairing_matrix is not None:
            degree = pairing_matrix.sum(-1, keepdim=True).clamp_min(1)
            adjacency = pairing_matrix / degree
            for layer in self.graph_layers:
                x = torch.relu(layer(torch.bmm(adjacency, x)) + x)
        x, _ = self.gru(x)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)

