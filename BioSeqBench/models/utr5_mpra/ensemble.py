"""Checkpoint-backed Optimus 5'UTR weak-surrogate ensembles."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from BioSeqBench.models.weakseqcnn import WeakSeqCNN


DNA_ALPHABET = "ACGT"


def encode_utr_sequences(sequences: Sequence[str] | np.ndarray) -> torch.Tensor:
    """Encode fixed-length DNA strings as contiguous [N, 4, L] float tensors."""
    values = np.asarray(sequences, dtype=str).reshape(-1)
    if len(values) == 0:
        return torch.empty((0, 4, 50), dtype=torch.float32)
    lengths = np.char.str_len(values)
    if np.any(lengths != 50):
        bad = int(np.flatnonzero(lengths != 50)[0])
        raise ValueError(f"Optimus requires 50-nt sequences; row {bad} has length {lengths[bad]}")
    raw = np.frombuffer("".join(values.tolist()).encode("ascii"), dtype=np.uint8).reshape(-1, 50)
    lookup = np.full(256, -1, dtype=np.int8)
    lookup[np.frombuffer(DNA_ALPHABET.encode("ascii"), dtype=np.uint8)] = np.arange(4, dtype=np.int8)
    encoded = lookup[raw]
    if np.any(encoded < 0):
        bad_row, bad_column = np.argwhere(encoded < 0)[0]
        raise ValueError(
            f"Invalid DNA base at row {int(bad_row)}, position {int(bad_column)}: "
            f"{values[int(bad_row)][int(bad_column)]!r}"
        )
    one_hot = np.eye(4, dtype=np.float32)[encoded].transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(one_hot))


def decode_utr_one_hot(one_hot: torch.Tensor) -> np.ndarray:
    """Decode [N, L, 4] or [N, 4, L] tensors by argmax."""
    array = one_hot.detach().cpu().numpy()
    if array.ndim != 3:
        raise ValueError("one_hot must be rank three")
    if array.shape[-1] == 4:
        indices = array.argmax(axis=-1)
    elif array.shape[1] == 4:
        indices = array.argmax(axis=1)
    else:
        raise ValueError("one_hot must have four DNA channels")
    alphabet = np.asarray(list(DNA_ALPHABET))
    return np.asarray(["".join(row) for row in alphabet[indices]], dtype=str)


class UTR5WeakEnsemble:
    """Load standardized lower-tail WeakSeqCNN-v1 checkpoints and predict MRL."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str | torch.device | None = None,
        member_indices: Sequence[int] | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        paths = sorted(self.checkpoint_dir.glob("member_*.pt"))
        if not paths:
            raise FileNotFoundError(f"No member_*.pt checkpoints in {self.checkpoint_dir}")
        if member_indices is not None:
            requested = {int(index) for index in member_indices}
            available = {int(path.stem.split("_")[-1]): path for path in paths}
            missing = requested - set(available)
            if missing:
                raise FileNotFoundError(
                    f"Missing weak-model members {sorted(missing)} in {self.checkpoint_dir}"
                )
            paths = [available[index] for index in sorted(requested)]
        self.checkpoint_paths = paths
        self.models: list[nn.Module] = []
        means, scales = [], []
        self.states: list[dict] = []
        for path in paths:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state.get("dataset") != "utr5_mpra" or int(state.get("sequence_length", -1)) != 50:
                raise ValueError(f"Incompatible UTR5 weak checkpoint: {path}")
            if state.get("model") != "weakseqcnn_v1":
                raise ValueError(
                    f"Incompatible UTR5 weak checkpoint architecture in {path}: "
                    f"expected weakseqcnn_v1, found {state.get('model')!r}"
                )
            model = WeakSeqCNN(**state.get("model_config", {}))
            model.load_state_dict(state["model_state"])
            model.to(self.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models.append(model)
            means.append(float(state["train_label_mean"]))
            scales.append(float(state["train_label_std"]))
            self.states.append(state)
        self.label_means = np.asarray(means, dtype=np.float32)
        self.label_stds = np.asarray(scales, dtype=np.float32)

    @torch.no_grad()
    def predict(
        self,
        sequences: Sequence[str] | np.ndarray,
        *,
        batch_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = encode_utr_sequences(sequences).transpose(1, 2).contiguous()
        loader = DataLoader(
            TensorDataset(features),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        member_outputs = [[] for _ in self.models]
        for (batch,) in loader:
            batch = batch.to(self.device, non_blocking=True)
            for index, model in enumerate(self.models):
                normalized = model(batch)
                restored = normalized * self.label_stds[index] + self.label_means[index]
                member_outputs[index].append(restored.cpu())
        matrix = np.stack([torch.cat(parts).numpy() for parts in member_outputs])
        return matrix.mean(axis=0), matrix.std(axis=0)


class DifferentiableUTR5Ensemble(nn.Module):
    """Frozen calibrated weak ensemble retaining gradients w.r.t. one-hot input."""

    def __init__(self, ensemble: UTR5WeakEnsemble) -> None:
        super().__init__()
        self.models = nn.ModuleList(ensemble.models)
        self.register_buffer("label_means", torch.tensor(ensemble.label_means))
        self.register_buffer("label_stds", torch.tensor(ensemble.label_stds))
        self.eval()

    def member_predictions(self, one_hot: torch.Tensor) -> torch.Tensor:
        if one_hot.ndim != 3:
            raise ValueError("one_hot must have shape [batch, length, 4] or [batch, 4, length]")
        features = one_hot if one_hot.shape[-1] == 4 else one_hot.transpose(1, 2)
        if features.shape[1:] != (50, 4):
            raise ValueError("5'UTR WeakSeqCNN input must contain 50 positions and four channels")
        predictions = []
        for index, model in enumerate(self.models):
            normalized = model(features)
            predictions.append(normalized * self.label_stds[index] + self.label_means[index])
        return torch.stack(predictions)

    def forward(self, one_hot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        members = self.member_predictions(one_hot)
        return members.mean(dim=0), members.std(dim=0, unbiased=False)
