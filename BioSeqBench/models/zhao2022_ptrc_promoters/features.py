"""Feature construction shared by the Ptrc XGBoost and CNN baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


BASES = "ACGT"
BASE_TO_CHANNEL = {base: index for index, base in enumerate(BASES)}


def one_hot_sequences(sequences: pd.Series, max_length: int = 76) -> np.ndarray:
    """Return zero-padded positional A/C/G/T one-hot encodings.

    Flattening this tensor gives the positional nucleotide representation used
    by the XGBoost reproduction.  Retaining its channel/length shape gives the
    CNN exactly the same information.
    """
    values = sequences.astype(str).str.upper().tolist()
    encoded = np.zeros((len(values), 4, max_length), dtype=np.float32)
    for row, sequence in enumerate(values):
        if len(sequence) > max_length:
            raise ValueError(f"Sequence length {len(sequence)} exceeds max_length={max_length}")
        for position, base in enumerate(sequence):
            channel = BASE_TO_CHANNEL.get(base)
            if channel is None:
                raise ValueError(f"Unexpected nucleotide {base!r} in Ptrc sequence")
            encoded[row, channel, position] = 1.0
    return encoded


def one_hot_features(sequences: pd.Series, max_length: int = 76) -> np.ndarray:
    """Flatten ``one_hot_sequences`` for tree-based regression."""
    return one_hot_sequences(sequences, max_length=max_length).reshape(len(sequences), -1)


def log10_activity(frame: pd.DataFrame) -> np.ndarray:
    """Paper response scale: log10 fluorescence/OD600 (Figure 4C)."""
    values = frame["label"].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Ptrc fluorescence/OD600 labels must be finite and positive")
    return np.log10(values)


def torch_one_hot(sequences: pd.Series, max_length: int = 76) -> torch.Tensor:
    return torch.from_numpy(one_hot_sequences(sequences, max_length=max_length))
