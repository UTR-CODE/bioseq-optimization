"""Exact lookup oracles for the exhaustive TFBind8 and TFBind10 landscapes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DNA_BASES = np.asarray(list("ACGT"))
DNA_TO_INDEX = {base: index for index, base in enumerate(DNA_BASES)}
SUPPORTED_DATASETS = {"tfbind8": 8, "tfbind10": 10}


def encode_sequences(sequences: Sequence[str], length: int) -> np.ndarray:
    """Encode A/C/G/T strings as uint8 base indices with shape ``[N, length]``."""
    encoded = np.empty((len(sequences), length), dtype=np.uint8)
    for row, sequence in enumerate(sequences):
        normalized = str(sequence).upper()
        if len(normalized) != length:
            raise ValueError(f"Expected length-{length} DNA, got {sequence!r}")
        try:
            encoded[row] = [DNA_TO_INDEX[base] for base in normalized]
        except KeyError as error:
            raise ValueError(f"Invalid DNA base {error.args[0]!r} in {sequence!r}") from error
    return encoded


def encoded_to_codes(encoded: np.ndarray) -> np.ndarray:
    """Convert base-index rows to their lexicographic base-4 integer codes."""
    encoded = np.asarray(encoded)
    if encoded.ndim != 2 or np.any(encoded > 3):
        raise ValueError("encoded DNA must have shape [N, length] with values in 0..3")
    powers = 4 ** np.arange(encoded.shape[1] - 1, -1, -1, dtype=np.int64)
    return encoded.astype(np.int64, copy=False) @ powers


def codes_to_encoded(codes: np.ndarray, length: int) -> np.ndarray:
    """Convert lexicographic base-4 codes to uint8 base-index rows."""
    remaining = np.asarray(codes, dtype=np.int64).reshape(-1).copy()
    if np.any(remaining < 0) or np.any(remaining >= 4**length):
        raise ValueError(f"codes must be in [0, {4**length})")
    encoded = np.empty((len(remaining), length), dtype=np.uint8)
    for position in range(length - 1, -1, -1):
        encoded[:, position] = remaining % 4
        remaining //= 4
    return encoded


def decode_sequences(encoded: np.ndarray) -> np.ndarray:
    """Decode a base-index matrix into an array of DNA strings."""
    encoded = np.asarray(encoded)
    if encoded.ndim != 2 or np.any(encoded > 3):
        raise ValueError("encoded DNA must have shape [N, length] with values in 0..3")
    return np.fromiter(
        ("".join(row) for row in DNA_BASES[encoded]),
        dtype=f"U{encoded.shape[1]}",
        count=len(encoded),
    )


@dataclass(frozen=True)
class TFBindLandscape:
    """Complete landscape stored in base-4 order for constant-time true scoring."""

    dataset: str
    length: int
    labels_by_code: np.ndarray

    def __post_init__(self) -> None:
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset {self.dataset!r}")
        expected_length = SUPPORTED_DATASETS[self.dataset]
        if self.length != expected_length:
            raise ValueError(f"{self.dataset} requires length {expected_length}, got {self.length}")
        expected = 4**self.length
        if np.asarray(self.labels_by_code).shape != (expected,):
            raise ValueError(f"Expected {expected} labels, got {np.asarray(self.labels_by_code).shape}")
        if not np.isfinite(self.labels_by_code).all():
            raise ValueError("Landscape labels must all be finite")

    @property
    def size(self) -> int:
        return len(self.labels_by_code)

    @property
    def encoded(self) -> np.ndarray:
        return codes_to_encoded(np.arange(self.size, dtype=np.int64), self.length)

    def ranked_codes(self) -> np.ndarray:
        """Return codes ordered by increasing activity, breaking ties by sequence."""
        codes = np.arange(self.size, dtype=np.int64)
        return np.lexsort((codes, self.labels_by_code))

    def score_codes(self, codes: Sequence[int] | np.ndarray) -> np.ndarray:
        codes_array = np.asarray(codes, dtype=np.int64)
        if np.any(codes_array < 0) or np.any(codes_array >= self.size):
            raise ValueError(f"Sequence codes must be in [0, {self.size})")
        return self.labels_by_code[codes_array]

    def score(self, sequences: Sequence[str]) -> np.ndarray:
        encoded = encode_sequences(sequences, self.length)
        return self.score_codes(encoded_to_codes(encoded))

    def sequences_for_codes(self, codes: Sequence[int] | np.ndarray) -> np.ndarray:
        return decode_sequences(codes_to_encoded(np.asarray(codes), self.length))


def _load_tfbind8(raw_path: Path) -> TFBindLandscape:
    frame = pd.read_parquet(raw_path, columns=["sequence", "e_score"])
    encoded = encode_sequences(frame["sequence"].astype(str).tolist(), 8)
    codes = encoded_to_codes(encoded)
    labels = frame["e_score"].to_numpy(dtype=np.float64)
    return _build_landscape("tfbind8", 8, codes, labels)


def _load_tfbind10(raw_path: Path) -> TFBindLandscape:
    with np.load(raw_path, allow_pickle=False) as archive:
        if not {"x", "y"}.issubset(archive.files):
            raise ValueError(f"{raw_path} must contain arrays 'x' and 'y'")
        encoded = np.asarray(archive["x"])
        labels = np.asarray(archive["y"], dtype=np.float64).reshape(-1)
    if encoded.ndim != 2 or encoded.shape[1] != 10 or np.any((encoded < 0) | (encoded > 3)):
        raise ValueError(f"Expected TFBind10 x with shape [N, 10] and values in 0..3, got {encoded.shape}")
    return _build_landscape("tfbind10", 10, encoded_to_codes(encoded), labels)


def _build_landscape(dataset: str, length: int, codes: np.ndarray, labels: np.ndarray) -> TFBindLandscape:
    expected = 4**length
    if len(codes) != expected or len(labels) != expected:
        raise ValueError(f"{dataset} must contain the complete {expected:,}-sequence landscape")
    if len(np.unique(codes)) != expected:
        raise ValueError(f"{dataset} contains missing or duplicate DNA sequences")
    labels_by_code = np.empty(expected, dtype=np.float64)
    labels_by_code[codes] = labels
    return TFBindLandscape(dataset=dataset, length=length, labels_by_code=labels_by_code)


def load_tfbind_landscape(dataset: str, raw_path: str | Path | None = None) -> TFBindLandscape:
    """Load an exhaustive TFBind landscape from its compact raw source."""
    normalized = dataset.lower()
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(SUPPORTED_DATASETS)}, got {dataset!r}")
    if raw_path is None:
        raw_path = (
            Path("BioSeqBench/raw_data/tfbind8/six6_ref_r1.parquet")
            if normalized == "tfbind8"
            else Path("BioSeqBench/raw_data/tfbind10/TFBind10-Exact-v0.npz")
        )
    path = Path(raw_path)
    return _load_tfbind8(path) if normalized == "tfbind8" else _load_tfbind10(path)
