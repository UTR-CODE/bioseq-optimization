"""DREAM promoter Autosome/LegNet example integration."""

from __future__ import annotations

import numpy as np

from BioSeqBench.models.yeast_gpra_dream2022 import LegNetAutosome
from BioSeqBench.training.train_yeast_dream2022_autosome import encode_batch

from ...api import TorchSequenceScorer


class _ScoreOnlyLegNet:
    def __init__(self, model: LegNetAutosome):
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, inputs):
        _, score = self.model(sequence_tensor=inputs)
        return score


class DreamLegNetExample(TorchSequenceScorer):
    """Wrap the DREAM LegNet model for 110/150-bp promoter strings."""

    def __init__(self, model: LegNetAutosome | None = None, **kwargs):
        def encoder(values):
            labels = np.zeros(len(values), dtype=np.float32)
            inputs, _, _ = encode_batch(values.tolist(), labels, np.random.default_rng(0), reverse_probability=0.0)
            return inputs

        # Users who need reverse-complement averaging can pass a custom model
        # wrapper implementing that inference policy before this adapter.
        super().__init__(_ScoreOnlyLegNet(model or LegNetAutosome()), encoder, **kwargs)
