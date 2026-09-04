"""Faithful MPRA-LegNet architecture from autosome-ru/human_legnet.

The model is a LegNet variant used by Agarwal et al. for 230-bp lentiMPRA
constructs.  Its input is A/G/C/T one-hot data plus an optional fifth channel
that identifies reverse-complement augmentation.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class _SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.SiLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate = self.fc(inputs.mean(dim=2)).unsqueeze(-1)
        return inputs * gate


class _EffBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, resize_factor: int) -> None:
        super().__init__()
        inner = channels * resize_factor
        self.block = nn.Sequential(
            nn.Conv1d(channels, inner, kernel_size=1, padding="same", bias=False),
            nn.BatchNorm1d(inner), nn.SiLU(),
            nn.Conv1d(inner, inner, kernel_size=kernel_size, groups=inner, padding="same", bias=False),
            nn.BatchNorm1d(inner), nn.SiLU(),
            _SELayer(inner, reduction=resize_factor),
            nn.Conv1d(inner, channels, kernel_size=1, padding="same", bias=False),
            nn.BatchNorm1d(channels), nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class _LegNetBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, resize_factor: int, pool_size: int) -> None:
        super().__init__()
        self.efficient = _EffBlock(input_channels, kernel_size, resize_factor)
        self.local = nn.Sequential(
            nn.Conv1d(2 * input_channels, output_channels, kernel_size=kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(output_channels), nn.SiLU(),
        )
        self.pool = nn.MaxPool1d(pool_size) if pool_size != 1 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.local(torch.cat([self.efficient(inputs), inputs], dim=1)))


class MPRALegNet(nn.Module):
    """Author MPRA-LegNet with the published default configuration."""

    def __init__(
        self,
        input_channels: int = 5,
        stem_channels: int = 64,
        stem_kernel_size: int = 11,
        efficient_kernel_size: int = 9,
        block_sizes: tuple[int, ...] = (80, 96, 112, 128),
        pool_sizes: tuple[int, ...] = (2, 2, 2, 2),
        resize_factor: int = 4,
    ) -> None:
        super().__init__()
        if len(block_sizes) != len(pool_sizes):
            raise ValueError("block_sizes and pool_sizes must have equal length")
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, kernel_size=stem_kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(stem_channels), nn.SiLU(),
        )
        current = stem_channels
        blocks = []
        for channels, pool_size in zip(block_sizes, pool_sizes):
            blocks.append(_LegNetBlock(current, channels, efficient_kernel_size, resize_factor, pool_size))
            current = channels
        self.blocks = nn.Sequential(*blocks)
        self.mapper = nn.Sequential(nn.BatchNorm1d(current), nn.Conv1d(current, 2 * current, kernel_size=1))
        self.head = nn.Sequential(
            nn.Linear(2 * current, 2 * current), nn.BatchNorm1d(2 * current), nn.SiLU(), nn.Linear(2 * current, 1),
        )

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        hidden = self.mapper(self.blocks(self.stem(sequence_tensor)))
        return self.head(F.adaptive_avg_pool1d(hidden, 1).squeeze(-1))


# The public ``best_model.zip`` checkpoints were saved through the original
# Lightning implementation.  Its module hierarchy (``stem.block`` and
# ``main.*``) is part of every state-dict key, so it needs this structurally
# identical, source-key-compatible implementation for direct loading.
class _OfficialLocalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class _OfficialEffBlock(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int, resize_factor: int) -> None:
        super().__init__()
        inner = in_ch * resize_factor
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, inner, kernel_size=1, padding="same", bias=False),
            nn.BatchNorm1d(inner), nn.SiLU(),
            nn.Conv1d(inner, inner, kernel_size=kernel_size, groups=inner, padding="same", bias=False),
            nn.BatchNorm1d(inner), nn.SiLU(),
            _SELayer(inner, reduction=resize_factor),
            nn.Conv1d(inner, in_ch, kernel_size=1, padding="same", bias=False),
            nn.BatchNorm1d(in_ch), nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class _OfficialResidualConcat(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.concat([self.fn(inputs), inputs], dim=1)


class _OfficialMapperBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(in_features, out_features, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class OfficialMPRALegNet(nn.Module):
    """Exact module naming/layout required by Agarwal et al. checkpoints."""

    def __init__(
        self,
        in_ch: int = 4,
        stem_ch: int = 64,
        stem_ks: int = 11,
        ef_ks: int = 9,
        ef_block_sizes: tuple[int, ...] = (80, 96, 112, 128),
        pool_sizes: tuple[int, ...] = (2, 2, 2, 2),
        resize_factor: int = 4,
    ) -> None:
        super().__init__()
        if len(pool_sizes) != len(ef_block_sizes):
            raise ValueError("pool_sizes and ef_block_sizes must have equal length")
        self.stem = _OfficialLocalBlock(in_ch, stem_ch, stem_ks)
        blocks: list[nn.Module] = []
        current = stem_ch
        for pool_size, out_ch in zip(pool_sizes, ef_block_sizes):
            blocks.append(nn.Sequential(
                _OfficialResidualConcat(_OfficialEffBlock(current, ef_ks, resize_factor)),
                _OfficialLocalBlock(current * 2, out_ch, ef_ks),
                nn.MaxPool1d(pool_size) if pool_size != 1 else nn.Identity(),
            ))
            current = out_ch
        self.main = nn.Sequential(*blocks)
        self.mapper = _OfficialMapperBlock(current, current * 2)
        self.head = nn.Sequential(
            nn.Linear(current * 2, current * 2), nn.BatchNorm1d(current * 2), nn.SiLU(), nn.Linear(current * 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.mapper(self.main(self.stem(inputs)))
        hidden = F.adaptive_avg_pool1d(hidden, 1).squeeze(-1)
        return self.head(hidden).squeeze(-1)


def initialise_mpralegnet_weights(module: nn.Module) -> None:
    """Exact initializer in the authors' ``model.py``."""
    if isinstance(module, nn.Conv1d):
        module.weight.data.normal_(0, math.sqrt(2 / (module.kernel_size[0] * module.out_channels)))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(0, 0.001)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
