"""Compose independent scorers without model-specific optimizer logic."""

from collections.abc import Sequence

import numpy as np

from .api import (
    ScoreBatch,
    SequenceScorer,
    as_score_batch,
    normalize_nucleotide_sequences,
)


class CombinedScorer:
    """Evaluate the same sequences with one single-output scorer per column.

    Each child owns its encoding, padding and context. Counters are cumulative
    sequence evaluations per child (including repeated calls), not API batches.
    Unknown uncertainty in any child leaves combined uncertainty unspecified.
    """

    def __init__(self, models: Sequence[SequenceScorer], *, names=None, device=None):
        self.models = tuple(models)
        if not self.models or any(
            not callable(getattr(model, "score", None)) for model in self.models
        ):
            raise ValueError(
                "models must contain at least one scorer implementing score"
            )
        self.task_names = (
            tuple(names)
            if names is not None
            else tuple(f"model_{i}" for i in range(len(self.models)))
        )
        if (
            len(self.task_names) != len(self.models)
            or any(not isinstance(name, str) or not name for name in self.task_names)
            or len(set(self.task_names)) != len(self.task_names)
        ):
            raise ValueError("names must contain one unique nonempty string per model")
        self.device = (
            device if device is not None else getattr(self.models[0], "device", None)
        )
        self.model_query_counts = dict.fromkeys(self.task_names, 0)
        self.model_soft_query_counts = dict.fromkeys(self.task_names, 0)

    @property
    def supports_soft_scoring(self) -> bool:
        """Whether every child accepts canonical A/C/G/T soft probabilities."""
        return all(
            callable(getattr(model, "score_soft", None))
            and bool(getattr(model, "supports_soft_scoring", True))
            for model in self.models
        )

    @staticmethod
    def _column(values, rows, name):
        if values.ndim == 1:
            values = values[:, None]
        if values.shape != (rows, 1):
            raise ValueError(
                f"model {name!r} must return [N] or [N, 1]; got {tuple(values.shape)}"
            )
        return values

    def score(self, sequences):
        values = normalize_nucleotide_sequences(sequences)
        scores, uncertainties = [], []
        for name, model in zip(self.task_names, self.models):
            batch = as_score_batch(model.score(values.copy()))
            self.model_query_counts[name] += len(values)
            scores.append(self._column(batch.scores, len(values), name))
            uncertainties.append(
                None
                if batch.uncertainties is None
                else self._column(batch.uncertainties, len(values), name)
            )
        return ScoreBatch(
            np.concatenate(scores, axis=1),
            None
            if any(value is None for value in uncertainties)
            else np.concatenate(uncertainties, axis=1),
        )

    def score_soft(self, probabilities):
        import torch

        missing = [
            name
            for name, model in zip(self.task_names, self.models)
            if not callable(getattr(model, "score_soft", None))
        ]
        if missing:
            raise TypeError(
                f"gradient optimization requires score_soft on every model; missing: {missing}"
            )
        if probabilities.ndim != 3:
            raise ValueError("soft input must have shape [N, length, alphabet]")
        scores, uncertainties = [], []
        for name, model in zip(self.task_names, self.models):
            device = getattr(model, "device", None)
            inputs = probabilities if device is None else probabilities.to(device)
            activity, uncertainty = model.score_soft(inputs.clone())
            self.model_soft_query_counts[name] += len(probabilities)
            if not isinstance(activity, torch.Tensor):
                raise TypeError(
                    f"model {name!r} score_soft must return PyTorch tensors"
                )
            activity = self._column(activity, len(probabilities), name)
            if not torch.isfinite(activity).all():
                raise ValueError(f"model {name!r} returned non-finite soft activities")
            scores.append(activity.to(probabilities.device))
            if uncertainty is not None:
                uncertainty = self._column(uncertainty, len(probabilities), name)
                if not torch.isfinite(uncertainty).all():
                    raise ValueError(
                        f"model {name!r} returned non-finite soft uncertainties"
                    )
            uncertainties.append(
                None if uncertainty is None else uncertainty.to(probabilities.device)
            )
        return (
            torch.cat(scores, dim=1),
            None
            if any(value is None for value in uncertainties)
            else torch.cat(uncertainties, dim=1),
        )
