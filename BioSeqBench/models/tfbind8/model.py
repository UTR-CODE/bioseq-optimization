"""PyTorch DeepBind regression model adapted to the complete TFBind8 landscape."""

from __future__ import annotations

import torch
from torch import nn


class TFBind8DeepBind(nn.Module):
    """DeepBind M2 regression architecture with reverse-complement symmetry.

    This follows the authors' ``cfg/regression/maxpool/M2.cfg``: 16 motif
    filters, bias, ReLU, max pooling, dropout, a 32-unit ReLU layer and a
    linear regression output.  DeepBind's released TF configuration used a
    16-nt motif scanner on longer PBM probes.  TFBind8 is an exhaustive 8-mer
    landscape, so the only necessary task-specific adaptation is an 8-nt valid
    motif scanner.  With an 8-nt input max-pooling has one position, but it is
    retained explicitly to preserve the architecture.

    The released DeepBind TF trainer augments every sequence with its reverse
    complement.  TFBind8 labels are exactly reverse-complement invariant; this
    implementation averages the two strand predictions, which is the
    deterministic inference counterpart of that augmentation.
    """

    def __init__(
        self,
        input_channels: int = 4,
        output_dim: int = 1,
        num_motifs: int = 16,
        motif_length: int = 8,
        hidden_units: int = 32,
        dropout: float = 0.25,
        rc_ensemble: bool = True,
    ) -> None:
        super().__init__()
        self.motif_length = motif_length
        self.rc_ensemble = rc_ensemble
        self.motif_scan = nn.Conv1d(input_channels, num_motifs, motif_length, bias=True)
        self.activation = nn.ReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Sequential(nn.Flatten(), nn.Linear(num_motifs, hidden_units), nn.ReLU())
        self.output = nn.Linear(hidden_units, output_dim)

    @staticmethod
    def reverse_complement_one_hot(sequence_tensor: torch.Tensor) -> torch.Tensor:
        """Reverse positions and exchange A<->T, C<->G in A/C/G/T channel order."""
        if sequence_tensor.ndim != 3 or sequence_tensor.shape[1] != 4:
            raise ValueError("Expected DNA one-hot tensor with shape [batch, 4, length]")
        return sequence_tensor[:, [3, 2, 1, 0], :].flip(-1)

    def _forward_strand(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        if sequence_tensor.shape[-1] < self.motif_length:
            raise ValueError(
                f"TFBind8DeepBind requires length >= {self.motif_length}, got {sequence_tensor.shape[-1]}"
            )
        features = self.max_pool(self.activation(self.motif_scan(sequence_tensor)))
        return self.output(self.hidden(self.dropout(features)))

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        prediction = self._forward_strand(sequence_tensor)
        if self.rc_ensemble:
            prediction = 0.5 * (prediction + self._forward_strand(self.reverse_complement_one_hot(sequence_tensor)))
        return prediction


class TFBindCNN(nn.Module):
    """Small length-agnostic CNN for TFBind8 and TFBind10 weak ensembles.

    Angermueller et al. (ICLR 2020) describe the neural candidate surrogate as
    an ensemble of ten networks with two convolutional layers and one fully
    connected layer.  The paper does not specify channel counts, kernel sizes,
    padding, or pooling, so this is an explicit, compact implementation choice
    rather than a claim of exact unpublished hyperparameter recovery.
    """

    def __init__(
        self,
        input_channels: int = 4,
        output_dim: int = 1,
        channels: tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        hidden_units: int = 128,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(channels[0], channels[1], kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.LazyLinear(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim),
        )

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.encoder(sequence_tensor))


# Backwards-compatible name used by the original TFBind8 experiment.
TFBind8DyNACNN = TFBindCNN
