"""Iterable canonical-CSV dataset for BioSeqBench tables too large for RAM."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .SequenceDataset import LOOP_VOCAB, STRUCTURE_VOCAB, VOCABULARIES


class StreamingSequenceDataset(IterableDataset):
    """Read canonical CSV rows lazily with deterministic multi-worker sharding."""

    def __init__(
        self, csv_file: str | Path, *, alphabet: str, max_length: int,
        label_columns: list[str], num_records: int | None = None,
        cell_type_vocabulary: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.csv_file = Path(csv_file)
        self.alphabet = alphabet.upper()
        self.max_length = int(max_length)
        self.label_columns = list(label_columns)
        self.num_records = num_records
        self.token_to_id = {char: index + 1 for index, char in enumerate(VOCABULARIES[self.alphabet])}
        self.cell_type_vocabulary = cell_type_vocabulary or {"": 0, "multi_task": 1}

    @property
    def num_channels(self) -> int:
        return len(VOCABULARIES[self.alphabet])

    def __len__(self) -> int:
        if self.num_records is None:
            raise TypeError("Streaming dataset length is not known")
        return self.num_records

    def _encode(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = sequence[:self.max_length]
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:len(sequence)] = True
        if self.alphabet in {"DNA", "RNA"}:
            tensor = torch.zeros((self.num_channels, self.max_length), dtype=torch.float32)
            for index, char in enumerate(sequence):
                tensor[self.token_to_id[char] - 1, index] = 1.0
        else:
            tensor = torch.zeros(self.max_length, dtype=torch.long)
            for index, char in enumerate(sequence):
                tensor[index] = self.token_to_id[char]
        return tensor, mask

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        with self.csv_file.open(newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                if row_index % worker_count != worker_id:
                    continue
                sequence = row["sequence"]
                tensor, mask = self._encode(sequence)
                canonical = sequence.replace("U", "T")
                cell = row.get("cell_type", "") or ""
                structure = row.get("structure", "") or ""
                loop = row.get("predicted_loop_type", "") or ""
                structure_tensor = torch.zeros(self.max_length, dtype=torch.long)
                loop_tensor = torch.zeros(self.max_length, dtype=torch.long)
                for index, char in enumerate(structure[:self.max_length]): structure_tensor[index] = STRUCTURE_VOCAB.get(char, 0)
                for index, char in enumerate(loop[:self.max_length]): loop_tensor[index] = LOOP_VOCAB.get(char, 0)
                yield {
                    "sequence_tensor": tensor, "mask": mask,
                    "label": torch.tensor([float(row[column]) for column in self.label_columns], dtype=torch.float32).squeeze(0) if len(self.label_columns) == 1 else torch.tensor([float(row[column]) for column in self.label_columns], dtype=torch.float32),
                    "sequence_features": torch.tensor([
                        len(sequence) / max(1, self.max_length),
                        (canonical.count("G") + canonical.count("C")) / max(1, len(sequence)),
                        canonical.count("ATG") / max(1, len(sequence)),
                    ], dtype=torch.float32),
                    "structure_tensor": structure_tensor, "loop_tensor": loop_tensor,
                    "cell_type_id": torch.tensor(self.cell_type_vocabulary.get(cell, 0), dtype=torch.long),
                    "metadata": {"sequence_id": row["sequence_id"], "length": int(row["length"]), "cell_type": cell},
                }
