"""Checkpoint-aware DeepSTARR scorer example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from BioSeqBench.models.deepstarr import DeepSTARRPaper

from ...api import DifferentiableTorchSequenceScorer


DEFAULT_DEEPSTARR_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "BioSeqBench/checkpoints/strong_deepstarr_paper_v1/deepstarr/deepstarr_paper/best.pt"
)


def encode_deepstarr_sequences(sequences) -> torch.Tensor:
    """Encode 249-bp A/C/G/T strings as contiguous ``[N, 4, 249]`` tensors."""
    values = np.asarray(sequences, dtype=str).reshape(-1)
    if not len(values):
        return torch.empty((0, 4, 249), dtype=torch.float32)
    lengths = np.char.str_len(values)
    if np.any(lengths != 249):
        row = int(np.flatnonzero(lengths != 249)[0])
        raise ValueError(
            f"DeepSTARR requires 249-bp sequences; row {row} has length {lengths[row]}"
        )
    raw = np.frombuffer(
        "".join(values.tolist()).encode("ascii"), dtype=np.uint8
    ).reshape(-1, 249)
    lookup = np.full(256, -1, dtype=np.int8)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4, dtype=np.int8)
    encoded = lookup[raw]
    if np.any(encoded < 0):
        row, column = np.argwhere(encoded < 0)[0]
        raise ValueError(
            f"invalid DNA base at row {int(row)}, position {int(column)}: "
            f"{values[int(row)][int(column)]!r}"
        )
    one_hot = np.eye(4, dtype=np.float32)[encoded].transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(one_hot))


class DeepSTARRExample(DifferentiableTorchSequenceScorer):
    """Wrap DeepSTARR's developmental and housekeeping activity heads."""

    def __init__(self, model: DeepSTARRPaper | None = None, **kwargs):
        kwargs.setdefault("task_names", ("developmental", "housekeeping"))
        super().__init__(
            model or DeepSTARRPaper(),
            encode_deepstarr_sequences,
            soft_input_transform=lambda values: values.transpose(1, 2),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_DEEPSTARR_CHECKPOINT,
        **kwargs,
    ) -> "DeepSTARRExample":
        """Load the collected two-task DeepSTARR checkpoint."""
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"DeepSTARR checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = DeepSTARRPaper()
        model.load_state_dict(state["model_state"])
        return cls(model=model, **kwargs)
