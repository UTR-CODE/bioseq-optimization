"""Standardized lightweight surrogate CNN for DNA/RNA optimization tasks.

The implementation follows ``weak_model_description.txt``.  RNA is normalized
to the shared A/C/G/T alphabet before encoding, and the network accepts soft
nucleotide probabilities so gradients can flow back to sequence logits.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DNA_ALPHABET = "ACGT"
BASE_TO_INDEX = {base: index for index, base in enumerate(DNA_ALPHABET)}


def normalize_nucleotide_sequence(sequence: str) -> str:
    """Uppercase a nucleotide sequence and map RNA U to DNA T."""
    normalized = str(sequence).upper().replace("U", "T")
    invalid = sorted(set(normalized) - set(DNA_ALPHABET))
    if invalid:
        raise ValueError(f"Unsupported nucleotide symbols {invalid} in {sequence!r}")
    if not normalized:
        raise ValueError("Nucleotide sequences must not be empty")
    return normalized


def encode_nucleotide_sequences(
    sequences: Sequence[str], *, pad_to_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft-compatible one-hot ``[B,L,4]`` features and a boolean mask."""
    normalized = [normalize_nucleotide_sequence(sequence) for sequence in sequences]
    if not normalized:
        length = int(pad_to_length or 0)
        return torch.empty((0, length, 4), dtype=torch.float32), torch.empty((0, length), dtype=torch.bool)
    maximum = max(map(len, normalized))
    padded_length = maximum if pad_to_length is None else int(pad_to_length)
    if padded_length < maximum:
        raise ValueError(f"pad_to_length={padded_length} is shorter than a sequence of length {maximum}")
    features = torch.zeros((len(normalized), padded_length, 4), dtype=torch.float32)
    mask = torch.zeros((len(normalized), padded_length), dtype=torch.bool)
    for row, sequence in enumerate(normalized):
        indices = torch.tensor([BASE_TO_INDEX[base] for base in sequence], dtype=torch.long)
        features[row, : len(sequence)] = F.one_hot(indices, num_classes=4).float()
        mask[row, : len(sequence)] = True
    return features, mask


class MaskedGroupNorm1d(nn.GroupNorm):
    """GroupNorm that excludes padded positions when a sequence mask is given."""

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None or bool(mask.all()):
            return super().forward(inputs)
        if mask.shape != (inputs.shape[0], inputs.shape[2]):
            raise ValueError("mask must have shape [batch, length]")
        batch, channels, length = inputs.shape
        grouped = inputs.reshape(batch, self.num_groups, channels // self.num_groups, length)
        valid = mask[:, None, None, :].to(dtype=inputs.dtype, device=inputs.device)
        denominator = (valid.sum(dim=(2, 3)) * (channels // self.num_groups)).clamp_min(1.0)
        mean = (grouped * valid).sum(dim=(2, 3)) / denominator
        centered = grouped - mean[:, :, None, None]
        variance = (centered.square() * valid).sum(dim=(2, 3)) / denominator
        normalized = centered * torch.rsqrt(variance[:, :, None, None] + self.eps)
        normalized = normalized.reshape(batch, channels, length)
        if self.affine:
            normalized = normalized * self.weight[None, :, None] + self.bias[None, :, None]
        return normalized * mask[:, None, :].to(normalized.dtype)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = MaskedGroupNorm1d(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = MaskedGroupNorm1d(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        normalization_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        branch = self.conv1(inputs)
        branch = self.dropout(F.gelu(self.norm1(branch, normalization_mask)))
        branch = self.conv2(branch)
        branch = self.dropout(F.gelu(self.norm2(branch, normalization_mask)))
        output = F.gelu(inputs + branch)
        return output * mask[:, None, :].to(output.dtype)


class WeakSeqCNN(nn.Module):
    """WeakSeqCNN-v1 standardized differentiable nucleotide regressor."""

    def __init__(
        self,
        alphabet_size: int = 4,
        channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4, 8),
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if alphabet_size != 4:
            raise ValueError("WeakSeqCNN-v1 requires the unified A/C/G/T alphabet_size=4")
        if channels != 64 or num_blocks != 4 or kernel_size != 5:
            raise ValueError("WeakSeqCNN-v1 fixes channels=64, num_blocks=4, and kernel_size=5")
        if tuple(dilations) != (1, 2, 4, 8) or hidden_dim != 128 or dropout != 0.1:
            raise ValueError("WeakSeqCNN-v1 fixes dilations=(1,2,4,8), hidden_dim=128, and dropout=0.1")
        self.model_config = {
            "alphabet_size": alphabet_size,
            "channels": channels,
            "num_blocks": num_blocks,
            "kernel_size": kernel_size,
            "dilations": list(dilations),
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }
        self.input_projection = nn.Conv1d(alphabet_size, channels, kernel_size=1)
        self.position_projection = nn.Conv1d(channels + 2, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualDilatedBlock(channels, kernel_size, dilation, dropout)
            for dilation in dilations
        )
        self.attention_score = nn.Linear(channels, 1)
        self.head = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _relative_positions(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if bool(mask.all()):
            forward = torch.linspace(0, 1, mask.shape[1], dtype=dtype, device=mask.device)
            forward = forward[None, :].expand(mask.shape[0], -1)
            return torch.stack((forward, 1.0 - forward), dim=1)
        valid = mask.to(dtype)
        ordinal = (valid.cumsum(dim=1) - 1).clamp_min(0)
        denominator = (valid.sum(dim=1, keepdim=True) - 1).clamp_min(1)
        forward = (ordinal / denominator) * valid
        reverse = (1.0 - forward) * valid
        return torch.stack((forward, reverse), dim=1)

    def forward(
        self,
        x: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        sequence_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x is None:
            x = sequence_tensor
        if x is None or x.ndim != 3:
            raise ValueError("x must have shape [batch, length, 4]")
        # ``sequence_tensor`` compatibility keeps existing benchmark loaders
        # working while the canonical public interface remains [B,L,4].
        if x.shape[-1] == 4:
            sequence = x
        elif x.shape[1] == 4:
            sequence = x.transpose(1, 2)
        else:
            raise ValueError("x must contain four nucleotide channels")
        if not sequence.is_floating_point():
            raise TypeError("WeakSeqCNN input must be floating point one-hot/probabilities")
        batch, length, _ = sequence.shape
        normalization_mask: torch.Tensor | None
        if mask is None:
            mask = torch.ones((batch, length), dtype=torch.bool, device=sequence.device)
            normalization_mask = None
        else:
            mask = mask.to(device=sequence.device, dtype=torch.bool)
            if mask.shape != (batch, length):
                raise ValueError("mask must have shape [batch, length]")
            if not bool(mask.any(dim=1).all()):
                raise ValueError("every sequence must contain at least one unmasked nucleotide")
            normalization_mask = mask
        channels_first = sequence.transpose(1, 2)
        hidden = self.input_projection(channels_first)
        positions = self._relative_positions(mask, hidden.dtype)
        hidden = self.position_projection(torch.cat((hidden, positions), dim=1))
        hidden = hidden * mask[:, None, :].to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, mask, normalization_mask)
        mask_float = mask[:, None, :].to(hidden.dtype)
        mean_pool = (hidden * mask_float).sum(dim=-1) / mask_float.sum(dim=-1).clamp_min(1)
        max_pool = hidden.masked_fill(~mask[:, None, :], torch.finfo(hidden.dtype).min).max(dim=-1).values
        transposed = hidden.transpose(1, 2)
        attention_logits = self.attention_score(transposed).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, torch.finfo(hidden.dtype).min)
        attention = torch.softmax(attention_logits, dim=1)
        attention_pool = (attention[:, :, None] * transposed).sum(dim=1)
        features = torch.cat((mean_pool, max_pool, attention_pool), dim=-1)
        return self.head(features).squeeze(-1)
