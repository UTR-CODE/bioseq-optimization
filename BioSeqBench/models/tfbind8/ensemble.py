"""Inference wrapper for trained TFBind weak-CNN ensembles."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from BioSeqBench.oracles.tfbind import encode_sequences

from .model import TFBindCNN


class TFBindWeakEnsemble:
    """Load one lower-tail checkpoint directory as an optimization surrogate."""

    def __init__(self, checkpoint_dir: str | Path, device: str | torch.device | None = None) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        checkpoint_paths = sorted(self.checkpoint_dir.glob("member_*.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No member_*.pt checkpoints found in {self.checkpoint_dir}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models: list[TFBindCNN] = []
        self.label_means: list[float] = []
        self.label_stds: list[float] = []
        reference: tuple[str, int, float] | None = None
        for checkpoint_path in checkpoint_paths:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            identity = (str(state["dataset"]), int(state["sequence_length"]), float(state["train_fraction"]))
            if reference is None:
                reference = identity
            elif identity != reference:
                raise ValueError(f"Mixed dataset/length/fraction checkpoints in {self.checkpoint_dir}")
            model = TFBindCNN()
            model(torch.zeros(1, 4, identity[1]))
            model.load_state_dict(state["model_state"])
            self.models.append(model.to(self.device).eval())
            self.label_means.append(float(state["train_label_mean"]))
            self.label_stds.append(float(state["train_label_std"]))
        assert reference is not None
        self.dataset, self.sequence_length, self.train_fraction = reference

    @property
    def ensemble_size(self) -> int:
        return len(self.models)

    @torch.no_grad()
    def predict(self, sequences: Sequence[str], batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Return ensemble predictive mean and member standard deviation."""
        encoded = encode_sequences(sequences, self.sequence_length)
        features = torch.from_numpy(np.ascontiguousarray(np.eye(4, dtype=np.float32)[encoded].transpose(0, 2, 1)))
        all_predictions = []
        for model, label_mean, label_std in zip(self.models, self.label_means, self.label_stds):
            member_batches = []
            for batch in DataLoader(features, batch_size=batch_size, shuffle=False, pin_memory=self.device.type == "cuda"):
                normalized = model(batch.to(self.device, non_blocking=True)).squeeze(-1)
                member_batches.append((normalized * label_std + label_mean).cpu())
            all_predictions.append(torch.cat(member_batches).numpy())
        matrix = np.stack(all_predictions)
        return matrix.mean(axis=0), matrix.std(axis=0)
