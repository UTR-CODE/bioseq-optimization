"""Canonical table schema and sequence validation helpers."""

from __future__ import annotations

from typing import Final

CANONICAL_COLUMNS: Final[list[str]] = [
    "sequence_id", "sequence", "alphabet", "length", "label", "label_type",
    "cell_type", "organism", "source_reference",
]

ALPHABET_CHARS: Final[dict[str, set[str]]] = {
    "DNA": set("ACGT"),
    "RNA": set("ACGU"),
    "PROTEIN": set("ACDEFGHIKLMNPQRSTVWY"),
}


def normalize_sequence(sequence: object, alphabet: str) -> str:
    seq = str(sequence).strip().upper().replace(" ", "")
    if alphabet.upper() == "RNA":
        seq = seq.replace("T", "U")
    return seq


def is_valid_sequence(sequence: str, alphabet: str) -> bool:
    chars = ALPHABET_CHARS[alphabet.upper()]
    return bool(sequence) and set(sequence) <= chars

