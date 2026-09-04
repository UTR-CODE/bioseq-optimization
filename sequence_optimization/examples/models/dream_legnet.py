"""DREAM promoter Autosome/LegNet example integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from BioSeqBench.models.yeast_gpra_dream2022 import LegNetAutosome
from BioSeqBench.training.train_yeast_dream2022_autosome import encode_batch

from ...api import TorchSequenceScorer


DEFAULT_DREAM_LEGNET_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "BioSeqBench/checkpoints/strong_yeast_dream2022_autosome_v3_validation_5k/"
    "yeast_gpra_dream2022/autosome_legnet/best_val.pt"
)


class _ScoreOnlyLegNet(nn.Module):
    def __init__(self, model: LegNetAutosome):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        _, forward_score = self.model(sequence_tensor=inputs)
        reverse_dna = inputs[:, :4, :][:, [3, 2, 1, 0], :].flip(dims=[2])
        reverse_inputs = torch.cat([reverse_dna, inputs[:, 4:5, :]], dim=1)
        _, reverse_score = self.model(sequence_tensor=reverse_inputs)
        return (forward_score + reverse_score) / 2


class DreamLegNetExample(TorchSequenceScorer):
    """Wrap the DREAM LegNet model for fixed-length 110-bp promoter strings."""

    def __init__(self, model: LegNetAutosome | None = None, **kwargs):
        def encoder(values):
            lengths = np.char.str_len(values)
            if np.any(lengths != 110):
                row = int(np.flatnonzero(lengths != 110)[0])
                raise ValueError(
                    f"DREAM LegNet requires 110-bp sequences; row {row} has "
                    f"length {int(lengths[row])}"
                )
            labels = np.zeros(len(values), dtype=np.float32)
            inputs, _, _ = encode_batch(
                values.tolist(),
                labels,
                np.random.default_rng(0),
                reverse_probability=0.0,
            )
            return inputs

        kwargs.setdefault("task_names", ("promoter_activity",))
        super().__init__(_ScoreOnlyLegNet(model or LegNetAutosome()), encoder, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_DREAM_LEGNET_CHECKPOINT,
        **kwargs,
    ) -> "DreamLegNetExample":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"DREAM LegNet checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = LegNetAutosome()
        model.load_state_dict(state["model_state"])
        return cls(model=model, **kwargs)
