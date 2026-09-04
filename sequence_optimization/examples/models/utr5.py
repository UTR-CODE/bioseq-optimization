"""5′UTR model examples using the standalone scorer shell."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from BioSeqBench.models.utr5_mpra import Optimus5PrimePaper
from BioSeqBench.models.weakseqcnn import WeakSeqCNN
from BioSeqBench.models.utr5_mpra.ensemble import encode_utr_sequences

from ...api import DifferentiableTorchSequenceScorer
from .common import RestoredTargetScale


DEFAULT_OPTIMUS_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "BioSeqBench/checkpoints/strong_v2_sklearn/utr5_mpra/optimus5prime/best.pt"
)


class WeakSeqCNNExample(DifferentiableTorchSequenceScorer):
    """Wrap the standardized WeakSeqCNN-v1 checkpoint/model."""

    def __init__(self, model: WeakSeqCNN | None = None, **kwargs):
        kwargs.setdefault("task_names", ("activity",))
        super().__init__(
            model or WeakSeqCNN(),
            lambda values: encode_utr_sequences(values).transpose(1, 2),
            soft_input_transform=lambda values: values,
            **kwargs,
        )


class Optimus5PrimeExample(DifferentiableTorchSequenceScorer):
    """Wrap the collected Optimus 5′UTR model as a generic scorer."""

    def __init__(self, model: nn.Module | None = None, **kwargs):
        kwargs.setdefault("task_names", ("mrl",))
        super().__init__(
            model or Optimus5PrimePaper(),
            lambda values: encode_utr_sequences(values),
            soft_input_transform=lambda values: values.transpose(1, 2),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_OPTIMUS_CHECKPOINT,
        **kwargs,
    ) -> "Optimus5PrimeExample":
        """Load the collected Optimus checkpoint, including target scaling."""
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Optimus checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        model: nn.Module = Optimus5PrimePaper()
        model.load_state_dict(state["model_state"])
        target_mean, target_std = state.get("target_mean"), state.get("target_std")
        if target_mean is not None and target_std is not None:
            model = RestoredTargetScale(model, target_mean, target_std)
        return cls(model=model, **kwargs)
