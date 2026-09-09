"""Seed-sequence loading and reproducible random seed construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .api import _sequence_space


def load_seed_sequences(
    path: str | Path, length: int, *, count: int | None = None, alphabet: str = "ACGT"
) -> np.ndarray:
    """Load one sequence per non-empty, non-comment line from a text file."""
    space = _sequence_space(length, alphabet)
    values = [
        line.strip().upper()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    result = space.validate(values)
    if len(np.unique(result)) != len(result):
        raise ValueError("seed file contains duplicate sequences")
    if count is not None and len(result) != count:
        raise ValueError(
            f"seed file contains {len(result)} sequences; expected {count}"
        )
    return result


def random_seed_sequences(
    length: int, count: int, *, seed: int = 0, alphabet: str = "ACGT"
) -> np.ndarray:
    """Create exactly ``count`` unique random seed sequences where feasible."""
    space = _sequence_space(length, alphabet)
    if count < 1:
        raise ValueError("count must be positive")
    capacity = space.alphabet_size**space.length
    if count > capacity:
        raise ValueError(
            f"cannot draw {count} unique sequences from a space of size {capacity}"
        )
    rng = np.random.default_rng(seed)
    values: list[str] = []
    seen: set[str] = set()
    while len(values) < count:
        for sequence in space.random(min(max(64, count - len(values)), 4096), rng):
            if sequence not in seen:
                seen.add(str(sequence))
                values.append(str(sequence))
                if len(values) == count:
                    break
    return np.asarray(values, dtype=str)
