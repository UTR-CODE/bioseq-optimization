"""TFBind8 model example integration."""

from __future__ import annotations

from BioSeqBench.models.tfbind8 import TFBindCNN
from BioSeqBench.oracles.tfbind import encode_sequences

from ...api import TorchSequenceScorer


class TFBind8Example(TorchSequenceScorer):
    """Wrap a TFBind8 CNN; ``length`` must match the checkpoint."""

    def __init__(self, model: TFBindCNN | None = None, *, length: int = 8, **kwargs):
        encoder = lambda values: __import__("torch").from_numpy(
            __import__("numpy").ascontiguousarray(
                __import__("numpy").eye(4, dtype="float32")[encode_sequences(values, length)].transpose(0, 2, 1)
            )
        )
        super().__init__(model or TFBindCNN(), encoder, **kwargs)
