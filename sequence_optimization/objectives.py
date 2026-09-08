"""Scalar objectives for single- and multi-task activity models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .api import ScoreBatch, SequenceScorer, as_score_batch


def _activity_matrix(values: np.ndarray, expected_rows: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) != expected_rows:
        raise ValueError(
            f"activity model must return shape [N] or [N, tasks]; received {array.shape}"
        )
    return array


@dataclass(frozen=True)
class TargetObjective:
    """Select and scalarize one output of an activity model.

    ``mode='maximize'`` maximizes only the selected output. In
    ``mode='specificity'``, the utility is the target activity minus a weighted
    penalty on all other outputs. The ``1 / num_tasks`` scaling matches
    CREsted's weighted-difference objective.
    """

    target: int | str = 0
    mode: str = "maximize"
    penalty: float = 1.0
    other_weights: Sequence[float] | Mapping[int | str, float] | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        aliases = {
            "maximize": "maximize",
            "target_only": "maximize",
            "specificity": "specificity",
            "target_specificity": "specificity",
            "crested": "specificity",
        }
        if mode not in aliases:
            raise ValueError("objective mode must be 'maximize' or 'specificity'")
        if not np.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError("objective penalty must be finite and non-negative")
        object.__setattr__(self, "mode", aliases[mode])

    def _target_index(self, n_tasks: int, task_names: tuple[str, ...] | None) -> int:
        if isinstance(self.target, str):
            if task_names is None:
                raise ValueError("a string target requires task_names on the scorer")
            if self.target not in task_names:
                raise ValueError(
                    f"unknown target {self.target!r}; choose from {task_names}"
                )
            return task_names.index(self.target)
        index = int(self.target)
        if not 0 <= index < n_tasks:
            raise ValueError(f"target index {index} is outside {n_tasks} model outputs")
        return index

    def coefficients(
        self, n_tasks: int, task_names: tuple[str, ...] | None = None
    ) -> np.ndarray:
        if n_tasks < 1:
            raise ValueError("an activity model must have at least one output")
        if task_names is not None and len(task_names) != n_tasks:
            raise ValueError("task_names must match the number of model outputs")
        target = self._target_index(n_tasks, task_names)
        coefficients = np.zeros(n_tasks, dtype=np.float64)
        coefficients[target] = 1.0
        if self.mode == "maximize" or n_tasks == 1 or self.penalty == 0:
            return coefficients

        weights = np.ones(n_tasks, dtype=np.float64)
        if isinstance(self.other_weights, Mapping):
            for key, value in self.other_weights.items():
                if isinstance(key, str):
                    if task_names is None or key not in task_names:
                        raise ValueError(f"unknown task weight key {key!r}")
                    index = task_names.index(key)
                else:
                    index = int(key)
                    if not 0 <= index < n_tasks:
                        raise ValueError(
                            f"task weight index {index} is outside {n_tasks} outputs"
                        )
                weights[index] = float(value)
        elif self.other_weights is not None:
            weights = np.asarray(self.other_weights, dtype=np.float64)
            if weights.shape != (n_tasks,):
                raise ValueError(
                    "other_weights must contain one value per model output"
                )
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("other task weights must be finite and non-negative")
        weights[target] = 0.0
        coefficients -= self.penalty * weights / n_tasks
        coefficients[target] = 1.0
        return coefficients

    def scalarize(
        self,
        activities: np.ndarray,
        task_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        values = np.asarray(activities, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("activities must have shape [N] or [N, tasks]")
        return values @ self.coefficients(values.shape[1], task_names)

    def describe(
        self, n_tasks: int, task_names: tuple[str, ...] | None = None
    ) -> dict[str, object]:
        target_index = self._target_index(n_tasks, task_names)
        return {
            "mode": self.mode,
            "target_index": target_index,
            "target_name": None if task_names is None else task_names[target_index],
            "penalty": self.penalty,
            "coefficients": self.coefficients(n_tasks, task_names).tolist(),
        }


class ObjectiveScorer:
    """Turn raw model activities into scalar utilities and retain activities."""

    def __init__(self, scorer: SequenceScorer, objective: TargetObjective) -> None:
        self.scorer = scorer
        self.objective = objective
        names = getattr(scorer, "task_names", None)
        self.task_names = None if names is None else tuple(names)
        self._activities: dict[str, np.ndarray] = {}

    def score(self, sequences) -> ScoreBatch:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        result = as_score_batch(self.scorer.score(values))
        activities = _activity_matrix(result.scores, len(values))
        if self.task_names is not None and len(self.task_names) != activities.shape[1]:
            raise ValueError("task_names must match the number of model outputs")
        utilities = self.objective.scalarize(activities, self.task_names)

        utility_uncertainties = None
        if result.uncertainties is not None:
            uncertainty = _activity_matrix(result.uncertainties, len(values))
            if uncertainty.shape != activities.shape:
                raise ValueError("activity uncertainties must match activity outputs")
            coefficients = self.objective.coefficients(
                activities.shape[1], self.task_names
            )
            utility_uncertainties = np.sqrt((uncertainty**2) @ (coefficients**2))

        for sequence, activity in zip(values, activities):
            self._activities[str(sequence)] = activity.copy()
        return ScoreBatch(utilities, utility_uncertainties)

    def score_soft(self, probabilities):
        """Scalarize differentiable raw activities without detaching gradients."""
        import torch

        score_soft = getattr(self.scorer, "score_soft", None)
        if score_soft is None:
            raise TypeError(
                "this algorithm requires a differentiable scorer with score_soft(probabilities)"
            )
        activities, uncertainties = score_soft(probabilities)
        if activities.ndim == 1:
            activities = activities[:, None]
        if self.task_names is not None and len(self.task_names) != activities.shape[1]:
            raise ValueError("task_names must match the number of model outputs")
        coefficients = torch.as_tensor(
            self.objective.coefficients(activities.shape[1], self.task_names),
            dtype=activities.dtype,
            device=activities.device,
        )
        utilities = activities @ coefficients
        utility_uncertainties = None
        if uncertainties is not None:
            utility_uncertainties = torch.sqrt(
                (uncertainties.square() * coefficients.square()).sum(dim=1)
            )
        return utilities, utility_uncertainties

    def activities_for(self, sequences) -> np.ndarray:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        missing = [str(value) for value in values if str(value) not in self._activities]
        if missing:
            raise KeyError(f"activities were not recorded for {len(missing)} sequences")
        return np.stack([self._activities[str(value)] for value in values])

    def resolved_task_names(self, n_tasks: int) -> tuple[str, ...]:
        if self.task_names is not None:
            return self.task_names
        return tuple(f"task_{index}" for index in range(n_tasks))
