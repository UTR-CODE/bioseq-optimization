"""Checkpoint-backed Zhou L3 promoter activity model."""

from __future__ import annotations

from pathlib import Path

import torch

from BioSeqBench.models.zhou2026_l3_promoters import L3PositionAwareCNN

from ...api import DifferentiableTorchSequenceScorer
from .common import RestoredTargetScale, encode_acgt_sequences


DEFAULT_ZHOU_L3_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "BioSeqBench/checkpoints/strong_zhou2026_l3_position_v2/"
    "zhou2026_l3_promoters/zhou_l3_cnn/best.pt"
)


class ZhouL3Example(DifferentiableTorchSequenceScorer):
    """Predict activity for fixed-length 29-nt L3 promoters."""

    def __init__(self, model=None, **kwargs):
        kwargs.setdefault("task_names", ("promoter_activity",))
        super().__init__(
            model or L3PositionAwareCNN(length=29),
            lambda values: encode_acgt_sequences(values, exact_length=29),
            soft_input_transform=lambda values: values.transpose(1, 2),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_ZHOU_L3_CHECKPOINT,
        **kwargs,
    ) -> "ZhouL3Example":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Zhou L3 checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = L3PositionAwareCNN(output_dim=1, length=29)
        model.load_state_dict(state["model_state"])
        mean, scale = state.get("target_mean"), state.get("target_std")
        if mean is not None and scale is not None:
            model = RestoredTargetScale(model, mean, scale)
        return cls(model=model, **kwargs)
