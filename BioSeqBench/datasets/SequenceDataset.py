"""Unified PyTorch dataset for DNA, RNA, and protein sequences."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


VOCABULARIES = {
    "DNA": "ACGT",
    "RNA": "ACGU",
    "PROTEIN": "ACDEFGHIKLMNPQRSTVWY",
}
STRUCTURE_VOCAB = {".": 1, "(": 2, ")": 3, "[": 4, "]": 5, "{": 6, "}": 7, "<": 8, ">": 9}
LOOP_VOCAB = {char: index + 1 for index, char in enumerate("BEHIMSX")}


class SequenceDataset(Dataset):
    """Load a canonical CSV and return padded sequence tensors and masks.

    DNA/RNA are one-hot encoded as ``[channels, length]``. Proteins are tokenized
    as ``[length]`` with 0 reserved for padding and residues numbered from 1.
    """

    def __init__(
        self,
        csv_file: str | Path,
        alphabet: str | None = None,
        max_length: int | None = None,
        label_columns: list[str] | tuple[str, ...] | None = None,
        cell_type_vocabulary: dict[str, int] | None = None,
    ) -> None:
        self.frame = pd.read_csv(csv_file)
        if self.frame.empty:
            raise ValueError(f"Dataset is empty: {csv_file}")
        inferred = str(self.frame["alphabet"].iloc[0]).upper()
        self.alphabet = (alphabet or inferred).upper()
        if self.alphabet not in VOCABULARIES:
            raise ValueError(f"Unsupported alphabet {self.alphabet}")
        self.max_length = max_length or int(self.frame["length"].max())
        self.label_columns = list(label_columns or ["label"])
        missing = set(self.label_columns) - set(self.frame.columns)
        if missing:
            raise ValueError(f"Missing target columns: {sorted(missing)}")
        self.token_to_id = {char: i + 1 for i, char in enumerate(VOCABULARIES[self.alphabet])}
        cells = self.frame.get("cell_type", pd.Series([""] * len(self.frame))).fillna("").astype(str)
        self.cell_type_vocabulary = cell_type_vocabulary or {
            cell: i for i, cell in enumerate(sorted(cells.unique()))
        }

    @property
    def num_channels(self) -> int:
        return len(VOCABULARIES[self.alphabet])

    def __len__(self) -> int:
        return len(self.frame)

    def _encode(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = sequence[: self.max_length]
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[: len(sequence)] = True
        if self.alphabet in {"DNA", "RNA"}:
            tensor = torch.zeros((self.num_channels, self.max_length), dtype=torch.float32)
            for index, char in enumerate(sequence):
                tensor[self.token_to_id[char] - 1, index] = 1.0
        else:
            tensor = torch.zeros(self.max_length, dtype=torch.long)
            for index, char in enumerate(sequence):
                tensor[index] = self.token_to_id[char]
        return tensor, mask

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        sequence = str(row.sequence)
        tensor, mask = self._encode(sequence)
        labels = torch.tensor([float(row[column]) for column in self.label_columns], dtype=torch.float32)
        if len(self.label_columns) == 1:
            labels = labels.squeeze(0)
        cell_type = "" if pd.isna(row.get("cell_type", "")) else str(row.get("cell_type", ""))
        # Three low-cost features used by Optimus-style heads.
        canonical = sequence.replace("U", "T")
        features = torch.tensor([
            len(sequence) / max(1, self.max_length),
            (canonical.count("G") + canonical.count("C")) / max(1, len(sequence)),
            canonical.count("ATG") / max(1, len(sequence)),
        ], dtype=torch.float32)
        structure = "" if pd.isna(row.get("structure", "")) else str(row.get("structure", ""))
        loop = "" if pd.isna(row.get("predicted_loop_type", "")) else str(row.get("predicted_loop_type", ""))
        structure_tensor = torch.zeros(self.max_length, dtype=torch.long)
        loop_tensor = torch.zeros(self.max_length, dtype=torch.long)
        for i, char in enumerate(structure[: self.max_length]):
            structure_tensor[i] = STRUCTURE_VOCAB.get(char, 0)
        for i, char in enumerate(loop[: self.max_length]):
            loop_tensor[i] = LOOP_VOCAB.get(char, 0)
        result = {
            "sequence_tensor": tensor,
            "mask": mask,
            "label": labels,
            "sequence_features": features,
            "structure_tensor": structure_tensor,
            "loop_tensor": loop_tensor,
            "cell_type_id": torch.tensor(self.cell_type_vocabulary[cell_type], dtype=torch.long),
            "metadata": {
                "sequence_id": str(row.sequence_id),
                "length": int(row.length),
                "cell_type": cell_type,
            },
        }
        if structure:
            result["pairing_matrix"] = self._pairing_matrix(structure)
        return result

    def _pairing_matrix(self, structure: str) -> torch.Tensor:
        matrix = torch.eye(self.max_length, dtype=torch.float32)
        pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
        stacks = {opening: [] for opening in pairs.values()}
        for index, char in enumerate(structure[: self.max_length]):
            if char in stacks:
                stacks[char].append(index)
            elif char in pairs and stacks[pairs[char]]:
                partner = stacks[pairs[char]].pop()
                matrix[index, partner] = matrix[partner, index] = 1.0
        return matrix
