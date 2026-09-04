"""Deterministic cleaning, feature computation, and splitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CANONICAL_COLUMNS, is_valid_sequence, normalize_sequence


def gc_content(seq: str) -> float:
    if not seq:
        return float("nan")
    return (seq.count("G") + seq.count("C")) / len(seq)


def sequence_entropy(seq: str) -> float:
    if not seq:
        return float("nan")
    counts = Counter(seq)
    return -sum((n / len(seq)) * math.log2(n / len(seq)) for n in counts.values())


def kmer_frequencies(seq: str, k: int = 2) -> str:
    total = max(0, len(seq) - k + 1)
    if total == 0:
        return "{}"
    counts = Counter(seq[i:i + k] for i in range(total))
    return json.dumps({key: value / total for key, value in sorted(counts.items())}, separators=(",", ":"))


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def clean_table(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    alphabet: str,
    label_type: str,
    organism: str,
    source_reference: str,
    kmer_size: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """Return a canonical table and auditable cleaning counters.

    Duplicate identity includes cell type, because one sequence measured in multiple
    cell types is a valid multi-task observation.
    """
    frame = frame.copy()
    required = {"sequence", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Loader did not provide required columns: {sorted(missing)}")
    n_raw = len(frame)
    frame["sequence"] = frame["sequence"].map(lambda x: normalize_sequence(x, alphabet))
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    invalid_sequence = ~frame["sequence"].map(lambda x: is_valid_sequence(x, alphabet))
    invalid_label = frame["label"].isna() | ~np.isfinite(frame["label"])
    frame = frame.loc[~invalid_sequence & ~invalid_label].copy()
    if "cell_type" not in frame:
        frame["cell_type"] = ""
    frame["cell_type"] = frame["cell_type"].fillna("").astype(str)
    # The same sequence measured in another cell type, organism, reporter,
    # assay, or explicitly identified source observation is a distinct
    # experimental observation.  Keep such independent replicates while still
    # removing true within-context duplicates.
    duplicate_keys = ["sequence", "cell_type"]
    duplicate_keys.extend(
        column for column in ("organism", "reporter", "assay", "source_observation_id")
        if column in frame and column not in duplicate_keys
    )
    duplicate_mask = frame.duplicated(duplicate_keys, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].reset_index(drop=True)
    frame.insert(0, "sequence_id", [f"{dataset_name}_{i:06d}" for i in range(1, len(frame) + 1)])
    frame["alphabet"] = alphabet.upper()
    frame["length"] = frame["sequence"].str.len()
    frame["label_type"] = label_type
    if "organism" in frame:
        frame["organism"] = frame["organism"].fillna(organism).astype(str)
    else:
        frame["organism"] = organism
    frame["source_reference"] = source_reference
    frame["gc_content"] = frame["sequence"].map(gc_content) if alphabet.upper() in {"DNA", "RNA"} else np.nan
    frame["sequence_entropy"] = frame["sequence"].map(sequence_entropy)
    frame[f"kmer_{kmer_size}_frequency"] = frame["sequence"].map(lambda s: kmer_frequencies(s, kmer_size))
    extras = [column for column in frame.columns if column not in CANONICAL_COLUMNS]
    frame = frame[CANONICAL_COLUMNS + extras]
    audit = {
        "raw_rows": n_raw,
        "invalid_sequence_rows": int(invalid_sequence.sum()),
        "invalid_or_missing_label_rows": int(invalid_label.sum()),
        "duplicate_rows_removed": duplicate_count,
        "duplicate_rate": duplicate_count / max(1, n_raw),
        "clean_rows": len(frame),
    }
    return frame, audit


def deterministic_split(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    group_column: str = "sequence",
) -> dict[str, pd.DataFrame]:
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Fractions must satisfy train>0, val>=0, train+val<1")
    fraction = frame[group_column].astype(str).map(lambda x: stable_fraction(x, seed))
    masks = {
        "train": fraction < train_fraction,
        "val": (fraction >= train_fraction) & (fraction < train_fraction + val_fraction),
        "test": fraction >= train_fraction + val_fraction,
    }
    return {name: frame.loc[mask].reset_index(drop=True) for name, mask in masks.items()}


def write_dataset(splits: dict[str, pd.DataFrame], output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
