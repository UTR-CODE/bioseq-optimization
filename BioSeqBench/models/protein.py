from __future__ import annotations

import math

import torch
from torch import nn


class ProteinCNN(nn.Module):
    def __init__(self, vocab_size: int = 21, output_dim: int = 1, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, 7, padding=3), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.embedding(sequence_tensor).transpose(1, 2)))


class ProteinTransformer(nn.Module):
    def __init__(self, vocab_size: int = 21, output_dim: int = 1, d_model: int = 128, layers: int = 3, heads: int = 4, max_length: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, max_length, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))

    def forward(self, sequence_tensor: torch.Tensor, mask: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        x = self.embedding(sequence_tensor) + self.position[:, : sequence_tensor.shape[1]]
        x = self.encoder(x, src_key_padding_mask=~mask)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)

