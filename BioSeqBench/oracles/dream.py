"""Strong Autosome/LegNet oracle for DREAM promoter sequences."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from BioSeqBench.models.yeast_gpra_dream2022 import LegNetAutosome
from BioSeqBench.training.train_yeast_dream2022_autosome import encode_batch


class DreamAutosomeOracle:
    """Score raw DREAM promoter strings with the frozen strong LegNet model."""

    def __init__(self, checkpoint_path: str | Path, *, device: str | torch.device | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model = LegNetAutosome().to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.state = state

    @torch.no_grad()
    def predict(self, sequences: Sequence[str] | np.ndarray, *, batch_size: int = 4096) -> np.ndarray:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        labels = np.zeros(len(values), dtype=np.float32)
        outputs: list[torch.Tensor] = []
        rng = np.random.default_rng(0)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            inputs, _, _ = encode_batch(values[start:stop].tolist(), labels[start:stop], rng, reverse_probability=0.0)
            inputs = inputs.to(self.device, non_blocking=True)
            _, forward = self.model(sequence_tensor=inputs)
            reverse_dna = inputs[:, :4, :][:, [3, 2, 1, 0], :].flip(dims=[2])
            reverse_inputs = torch.cat([reverse_dna, inputs[:, 4:5, :]], dim=1)
            _, reverse = self.model(sequence_tensor=reverse_inputs)
            outputs.append(((forward + reverse) / 2).cpu())
        return torch.cat(outputs).numpy().astype(np.float64)
