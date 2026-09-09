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


@dataclass(frozen=True)
class MultiTaskObjective:
    """Optimize signed, optionally thresholded outputs of one model.

    Bounds and centers use raw model units; scales are fixed positive reference
    scales. Unselected tasks are ignored. ``worst`` maximizes the smallest
    margin; ``smoothmin`` uses the same smooth objective in NumPy and PyTorch.
    """

    high: Sequence[int | str] | str = ()
    low: Sequence[int | str] = ()
    aggregation: str = "worst"
    lower_bounds: Mapping[int | str, float] | None = None
    upper_bounds: Mapping[int | str, float] | None = None
    centers: Sequence[float] | None = None
    scales: Sequence[float] | None = None
    temperature: float = 0.1

    def __post_init__(self):
        if self.aggregation not in {"worst", "mean", "smoothmin"}:
            raise ValueError("aggregation must be worst, mean, or smoothmin")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    def _resolve(self, n_tasks, task_names):
        if n_tasks < 1 or (task_names is not None and len(task_names) != n_tasks):
            raise ValueError("task_names must match nonempty model outputs")

        def index(key):
            if isinstance(key, str):
                if task_names is None or key not in task_names:
                    raise ValueError(
                        f"unknown task {key!r}; named tasks require task_names"
                    )
                return task_names.index(key)
            if (
                isinstance(key, bool)
                or not isinstance(key, (int, np.integer))
                or not 0 <= key < n_tasks
            ):
                raise ValueError(f"invalid task index {key!r}")
            return int(key)

        high = (
            list(range(n_tasks))
            if isinstance(self.high, str) and self.high == "all"
            else [
                index(k)
                for k in ([self.high] if isinstance(self.high, str) else self.high)
            ]
        )
        low = [
            index(k) for k in ([self.low] if isinstance(self.low, str) else self.low)
        ]
        selected = high + low
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("select nonempty, distinct, disjoint high and low tasks")
        center = (
            np.zeros(n_tasks)
            if self.centers is None
            else np.asarray(self.centers, dtype=float)
        )
        scale = (
            np.ones(n_tasks)
            if self.scales is None
            else np.asarray(self.scales, dtype=float)
        )
        if (
            center.shape != (n_tasks,)
            or scale.shape != (n_tasks,)
            or not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError(
                "centers/scales must be finite per-task vectors with positive scales"
            )
        bounded = self.lower_bounds is not None or self.upper_bounds is not None
        if bounded:
            lower = {index(k): float(v) for k, v in (self.lower_bounds or {}).items()}
            upper = {index(k): float(v) for k, v in (self.upper_bounds or {}).items()}
            if set(lower) != set(high) or set(upper) != set(low):
                raise ValueError(
                    "bounds must cover exactly the selected high/low tasks"
                )
            center = center.copy()
            for k, value in {**lower, **upper}.items():
                if not np.isfinite(value):
                    raise ValueError("bounds must be finite")
                center[k] = value
        signs = np.asarray([1.0] * len(high) + [-1.0] * len(low))
        return selected, signs, center, scale, bounded

    def margins(self, activities, task_names=None):
        values = np.asarray(activities, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("activities must be a finite [N, tasks] matrix")
        ids, signs, center, scale, _ = self._resolve(values.shape[1], task_names)
        return (values[:, ids] - center[ids]) / scale[ids] * signs

    def scalarize(self, activities, task_names=None):
        margins = self.margins(activities, task_names)
        if self.aggregation == "mean":
            return margins.mean(axis=1)
        worst = margins.min(axis=1)
        if self.aggregation == "worst":
            return worst
        return worst - self.temperature * np.log(
            np.exp(-(margins - worst[:, None]) / self.temperature).mean(axis=1)
        )

    def scalarize_soft(self, activities, task_names=None):
        import torch

        ids, signs, center, scale, _ = self._resolve(activities.shape[1], task_names)

        def tensor(value):
            return torch.as_tensor(
                value, dtype=activities.dtype, device=activities.device
            )

        margins = (
            (activities[:, ids] - tensor(center[ids]))
            / tensor(scale[ids])
            * tensor(signs)
        )
        if self.aggregation == "mean":
            return margins.mean(dim=1)
        if self.aggregation == "worst":
            return margins.amin(dim=1)
        return -self.temperature * (
            torch.logsumexp(-margins / self.temperature, dim=1) - np.log(len(ids))
        )

    def uncertainty(self, activities, uncertainties, task_names=None):
        # First-order independent-output approximation, not joint confidence.
        margins = self.margins(activities, task_names)
        ids, _, _, scale, _ = self._resolve(np.asarray(activities).shape[1], task_names)
        if self.aggregation == "mean":
            weights = np.ones_like(margins) / margins.shape[1]
        elif self.aggregation == "worst":
            weights = (margins == margins.min(axis=1, keepdims=True)).astype(float)
            weights /= weights.sum(axis=1, keepdims=True)
        else:
            weights = np.exp(
                -(margins - margins.min(axis=1, keepdims=True)) / self.temperature
            )
            weights /= weights.sum(axis=1, keepdims=True)
        return np.sqrt(
            ((np.asarray(uncertainties)[:, ids] / scale[ids] * weights) ** 2).sum(
                axis=1
            )
        )

    def describe(self, n_tasks, task_names=None):
        ids, signs, center, scale, bounded = self._resolve(n_tasks, task_names)
        return {
            "mode": "multi_task",
            "aggregation": self.aggregation,
            "task_indices": ids,
            "directions": signs.tolist(),
            "centers_or_bounds": center.tolist(),
            "scales": scale.tolist(),
            "thresholded": bounded,
            "temperature": self.temperature,
            "uncertainty_method": "independent_first_order",
        }

    def evaluate(self, activities, task_names=None):
        ids, _, _, _, bounded = self._resolve(activities.shape[1], task_names)
        margins = self.margins(activities, task_names)
        return {
            "task_names": [task_names[i] if task_names else f"task_{i}" for i in ids],
            "margins": margins.tolist(),
            "worst_margin": margins.min(axis=1).tolist(),
            "satisfied": (margins >= 0).tolist() if bounded else None,
            "all_satisfied": (margins >= 0).all(axis=1).tolist() if bounded else None,
        }


class ObjectiveScorer:
    """Turn raw model activities into scalar utilities and retain activities."""

    def __init__(
        self, scorer: SequenceScorer, objective: TargetObjective | MultiTaskObjective
    ) -> None:
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
            if isinstance(self.objective, MultiTaskObjective):
                utility_uncertainties = self.objective.uncertainty(
                    activities, uncertainty, self.task_names
                )
            else:
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
        if isinstance(self.objective, MultiTaskObjective):
            utilities = self.objective.scalarize_soft(activities, self.task_names)
            uncertainty = None
            if uncertainties is not None:
                if uncertainties.shape != activities.shape:
                    raise ValueError(
                        "activity uncertainties must match activity outputs"
                    )
                ids, signs, center, scale, _ = self.objective._resolve(
                    activities.shape[1], self.task_names
                )

                def tensor(value):
                    return torch.as_tensor(
                        value, dtype=activities.dtype, device=activities.device
                    )

                margins = (
                    (activities[:, ids] - tensor(center[ids]))
                    / tensor(scale[ids])
                    * tensor(signs)
                )
                if self.objective.aggregation == "mean":
                    weights = torch.ones_like(margins) / len(ids)
                elif self.objective.aggregation == "worst":
                    weights = (margins == margins.amin(dim=1, keepdim=True)).to(
                        activities.dtype
                    )
                    weights = weights / weights.sum(dim=1, keepdim=True)
                else:
                    weights = torch.softmax(
                        -margins / self.objective.temperature, dim=1
                    )
                uncertainty = torch.linalg.vector_norm(
                    uncertainties[:, ids] / tensor(scale[ids]) * weights, dim=1
                )
            return utilities, uncertainty
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
