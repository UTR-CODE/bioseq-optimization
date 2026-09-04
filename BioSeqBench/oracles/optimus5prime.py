"""Frozen full-data Optimus 5'UTR strong-model oracle."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from BioSeqBench.models.utr5_mpra import Optimus5PrimePaper, encode_utr_sequences


DEFAULT_STRONG_CHECKPOINT = Path(
    "BioSeqBench/checkpoints/strong_test_guided_v1/utr5_mpra/optimus5prime/best.pt"
)


class Optimus5PrimeOracle:
    """Score arbitrary 50-nt DNA designs in restored MRL units."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_STRONG_CHECKPOINT,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        mean = state.get("standard_scaler_mean", state.get("target_mean"))
        scale = state.get("standard_scaler_scale", state.get("target_std"))
        if mean is None or scale is None:
            raise ValueError(f"Strong checkpoint lacks target calibration: {self.checkpoint_path}")
        self.mean = float(np.asarray(mean).reshape(-1)[0])
        self.scale = float(np.asarray(scale).reshape(-1)[0])
        self.model = Optimus5PrimePaper(output_dim=1)
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.state = state

    @torch.no_grad()
    def predict(
        self,
        sequences: Sequence[str] | np.ndarray,
        *,
        batch_size: int = 2048,
    ) -> np.ndarray:
        features = encode_utr_sequences(sequences)
        outputs = []
        loader = DataLoader(
            features,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        for batch in loader:
            normalized = self.model(sequence_tensor=batch.to(self.device, non_blocking=True)).squeeze(-1)
            outputs.append((normalized * self.scale + self.mean).cpu())
        return torch.cat(outputs).numpy().astype(np.float64)
