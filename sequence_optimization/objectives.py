"""Scalar objectives for single- and multi-task activity models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real

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
    scales. Unselected tasks are ignored. ``mean`` is the default equal-weight
    average, ``worst`` maximizes the smallest margin, and ``smoothmin``
    uses the same smooth objective in NumPy and PyTorch.
    """

    high: Sequence[int | str] | str = ()
    low: Sequence[int | str] = ()
    aggregation: str = "mean"
    lower_bounds: Mapping[int | str, float] | None = None
    upper_bounds: Mapping[int | str, float] | None = None
    centers: Sequence[float] | None = None
    scales: Sequence[float] | None = None
    temperature: float = 0.1
    directions: Sequence[str | float] | None = None
    weights: Sequence[float] | None = None

    def __post_init__(self):
        if self.directions is not None:
            if isinstance(self.directions, str) or not self.directions:
                raise ValueError(
                    "directions must be a nonempty list of numeric targets, "
                    "maximize, minimize, or ignore"
                )
            directions = []
            for value in self.directions:
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if normalized not in {"maximize", "minimize", "ignore"}:
                        raise ValueError(
                            "directions entries must be finite numeric targets, "
                            "maximize, minimize, or ignore"
                        )
                    directions.append(normalized)
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                ):
                    raise ValueError(
                        "directions entries must be finite numeric targets, "
                        "maximize, minimize, or ignore"
                    )
                else:
                    directions.append(float(value))
            if all(value == "ignore" for value in directions):
                raise ValueError("at least one output must be optimized")
            if len(self.high) or len(self.low):
                raise ValueError("use directions or high/low, not both")
            object.__setattr__(self, "directions", tuple(directions))
        if self.aggregation not in {"worst", "mean", "smoothmin"}:
            raise ValueError("aggregation must be worst, mean, or smoothmin")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.ndim != 1 or not len(weights):
                raise ValueError("weights must be a nonempty per-output vector")
            if not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("weights must be finite and non-negative")
            if weights.sum() <= 0:
                raise ValueError("at least one weight must be positive")
            object.__setattr__(self, "weights", tuple(float(w) for w in weights))

    def _resolve(self, n_tasks, task_names):
        if self.directions is not None and len(self.directions) != n_tasks:
            raise ValueError(
                "directions must contain exactly one entry per model output"
            )
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
        if self.directions is not None:
            high = [
                i
                for i, value in enumerate(self.directions)
                if value == "maximize" or not isinstance(value, str)
            ]
            low = [i for i, value in enumerate(self.directions) if value == "minimize"]
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
                "centers and scales must be finite per-task vectors with "
                "positive scales"
            )
        bounded = self.lower_bounds is not None or self.upper_bounds is not None
        direction_targets = self.target_thresholds(n_tasks)
        if direction_targets and bounded:
            raise ValueError(
                "numeric objective targets cannot be combined with lower_bounds "
                "or upper_bounds"
            )
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
        elif direction_targets:
            center = center.copy()
            for k, value in direction_targets.items():
                center[k] = value
            bounded = True
        signs = np.asarray([1.0] * len(high) + [-1.0] * len(low))
        return selected, signs, center, scale, bounded

    def target_thresholds(self, n_tasks: int | None = None) -> dict[int, float]:
        """Return numeric direction-list entries as raw-output lower targets."""
        if self.directions is None:
            return {}
        if n_tasks is not None and len(self.directions) != n_tasks:
            raise ValueError(
                "directions must contain exactly one entry per model output"
            )
        return {
            index: float(value)
            for index, value in enumerate(self.directions)
            if not isinstance(value, str)
        }

    def has_open_objectives(self) -> bool:
        return bool(
            self.directions
            and any(value in {"maximize", "minimize"} for value in self.directions)
        )

    def _active_positions(
        self,
        ids: Sequence[int],
        reached_targets: set[int] | frozenset[int] | None,
    ) -> np.ndarray:
        reached = set() if reached_targets is None else set(reached_targets)
        return np.asarray(
            [position for position, task in enumerate(ids) if task not in reached],
            dtype=int,
        )

    def _active_weights(
        self, ids: Sequence[int], active: np.ndarray, n_tasks: int
    ) -> np.ndarray | None:
        """Normalized weights over the active margin positions, or None if uniform."""
        if self.weights is None:
            return None
        if len(self.weights) != n_tasks:
            raise ValueError("weights must contain exactly one entry per model output")
        if self.aggregation == "worst":
            raise ValueError(
                "weights cannot be combined with aggregation='worst'; the minimum "
                "margin is invariant to positive weights, use 'mean' or 'smoothmin'"
            )
        selected = np.asarray([self.weights[ids[position]] for position in active])
        total = selected.sum()
        if total <= 0:
            raise ValueError("at least one active output must have a positive weight")
        return selected / total

    def margins(self, activities, task_names=None):
        values = np.asarray(activities, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("activities must be a finite [N, tasks] matrix")
        ids, signs, center, scale, _ = self._resolve(values.shape[1], task_names)
        return (values[:, ids] - center[ids]) / scale[ids] * signs

    def scalarize(self, activities, task_names=None, reached_targets=None):
        margins = self.margins(activities, task_names)
        ids, _, _, _, _ = self._resolve(np.asarray(activities).shape[1], task_names)
        active = self._active_positions(ids, reached_targets)
        if not len(active):
            return np.zeros(len(margins), dtype=np.float64)
        margins = margins[:, active]
        weights = self._active_weights(ids, active, np.asarray(activities).shape[1])
        if self.aggregation == "mean":
            if weights is None:
                return margins.mean(axis=1)
            return margins @ weights
        worst = margins.min(axis=1)
        if self.aggregation == "worst":
            return worst
        shifted = np.exp(-(margins - worst[:, None]) / self.temperature)
        pooled = (
            shifted.mean(axis=1) if weights is None else shifted @ weights
        )
        return worst - self.temperature * np.log(pooled)

    def scalarize_soft(self, activities, task_names=None, reached_targets=None):
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
        active = self._active_positions(ids, reached_targets)
        if not len(active):
            return activities[:, 0] * 0.0
        margins = margins[:, active]
        weights = self._active_weights(ids, active, activities.shape[1])
        if self.aggregation == "mean":
            if weights is None:
                return margins.mean(dim=1)
            return margins @ tensor(weights)
        if self.aggregation == "worst":
            return margins.amin(dim=1)
        if weights is None:
            return -self.temperature * (
                torch.logsumexp(-margins / self.temperature, dim=1)
                - np.log(len(active))
            )
        # Weighted smoothmin: log-sum-exp with per-output log-weight offsets.
        log_weights = torch.log(tensor(weights).clamp_min(1e-12))
        return -self.temperature * torch.logsumexp(
            -margins / self.temperature + log_weights, dim=1
        )

    def uncertainty(
        self, activities, uncertainties, task_names=None, reached_targets=None
    ):
        # First-order independent-output approximation, not joint confidence.
        margins = self.margins(activities, task_names)
        ids, _, _, scale, _ = self._resolve(np.asarray(activities).shape[1], task_names)
        active = self._active_positions(ids, reached_targets)
        if not len(active):
            return np.zeros(len(margins), dtype=np.float64)
        margins = margins[:, active]
        objective_weights = self._active_weights(
            ids, active, np.asarray(activities).shape[1]
        )
        ids = np.asarray(ids)[active].tolist()
        if self.aggregation == "mean":
            row = (
                np.full(len(active), 1 / len(active))
                if objective_weights is None
                else objective_weights
            )
            aggregation_weights = np.broadcast_to(row, margins.shape)
        elif self.aggregation == "worst":
            aggregation_weights = (
                margins == margins.min(axis=1, keepdims=True)
            ).astype(float)
            aggregation_weights /= aggregation_weights.sum(axis=1, keepdims=True)
        else:
            aggregation_weights = np.exp(
                -(margins - margins.min(axis=1, keepdims=True)) / self.temperature
            )
            if objective_weights is not None:
                aggregation_weights = aggregation_weights * objective_weights
            aggregation_weights /= aggregation_weights.sum(axis=1, keepdims=True)
        return np.sqrt(
            (
                (np.asarray(uncertainties)[:, ids] / scale[ids] * aggregation_weights)
                ** 2
            ).sum(axis=1)
        )

    def describe(self, n_tasks, task_names=None):
        ids, signs, center, scale, bounded = self._resolve(n_tasks, task_names)
        return {
            "mode": "multi_task",
            "aggregation": self.aggregation,
            "task_indices": ids,
            "directions": signs.tolist(),
            "requested_directions": (
                None if self.directions is None else list(self.directions)
            ),
            "centers_or_bounds": center.tolist(),
            "scales": scale.tolist(),
            "thresholded": bounded,
            "temperature": self.temperature,
            "uncertainty_method": "independent_first_order",
            "target_thresholds": {
                (task_names[index] if task_names else f"task_{index}"): value
                for index, value in self.target_thresholds(n_tasks).items()
            },
        }

    def evaluate(self, activities, task_names=None):
        ids, _, _, _, bounded = self._resolve(activities.shape[1], task_names)
        margins = self.margins(activities, task_names)
        targets = self.target_thresholds(activities.shape[1])
        if targets:
            positions = [ids.index(index) for index in targets]
            evaluated_ids = list(targets)
            evaluated_margins = margins[:, positions]
        else:
            evaluated_ids = ids
            evaluated_margins = margins
        satisfaction = evaluated_margins >= 0
        return {
            "task_names": [
                task_names[i] if task_names else f"task_{i}" for i in evaluated_ids
            ],
            "margins": evaluated_margins.tolist(),
            "worst_margin": evaluated_margins.min(axis=1).tolist(),
            "satisfied": satisfaction.tolist() if bounded else None,
            "all_satisfied": satisfaction.all(axis=1).tolist() if bounded else None,
        }


class ObjectiveScorer:
    """Turn raw model activities into scalar utilities and retain activities."""

    def __init__(
        self,
        scorer: SequenceScorer,
        objective: TargetObjective
        | MultiTaskObjective
        | list[str | float]
        | tuple[str | float, ...],
        weights: Sequence[float] | None = None,
    ) -> None:
        self.scorer = scorer
        if isinstance(objective, (list, tuple)):
            self.objective = MultiTaskObjective(
                directions=objective,
                aggregation="mean",
                weights=weights,
            )
        else:
            if weights is not None:
                if not isinstance(objective, MultiTaskObjective):
                    raise TypeError(
                        "weights require a directions list or MultiTaskObjective"
                    )
                objective = replace(objective, weights=weights)
            self.objective = objective
        names = getattr(scorer, "task_names", None)
        self.task_names = None if names is None else tuple(names)
        self._activities: dict[str, np.ndarray] = {}
        self._activity_uncertainties: dict[str, np.ndarray] = {}
        self._reached_targets: set[int] = set()
        self._objective_version = 0

    @property
    def objective_version(self) -> int:
        return self._objective_version

    def _targets(self, n_tasks: int) -> dict[int, float]:
        if not isinstance(self.objective, MultiTaskObjective):
            return {}
        return self.objective.target_thresholds(n_tasks)

    def _update_targets(self, activities: np.ndarray) -> None:
        targets = self._targets(activities.shape[1])
        if not targets:
            return
        eligible = np.ones(len(activities), dtype=bool)
        for index in self._reached_targets:
            eligible &= activities[:, index] >= targets[index]
        newly_reached: set[int] = set()
        for index, target in targets.items():
            if index in self._reached_targets:
                continue
            reaches_target = activities[:, index] >= target
            if np.any(eligible & reaches_target):
                newly_reached.add(index)
                eligible &= reaches_target
        if newly_reached:
            self._reached_targets.update(newly_reached)
            self._objective_version += 1

    def _activities_array(self) -> np.ndarray:
        if not self._activities:
            return np.empty((0, 0), dtype=np.float64)
        return np.stack(list(self._activities.values()))

    def _scalarize(self, activities: np.ndarray) -> np.ndarray:
        if isinstance(self.objective, MultiTaskObjective):
            return self.objective.scalarize(
                activities,
                self.task_names,
                reached_targets=self._reached_targets,
            )
        return self.objective.scalarize(activities, self.task_names)

    def _apply_target_gate(
        self, activities: np.ndarray, utilities: np.ndarray
    ) -> np.ndarray:
        targets = self._targets(activities.shape[1])
        if not self._reached_targets:
            return utilities
        reached = sorted(self._reached_targets)
        passed = np.ones(len(activities), dtype=bool)
        gaps = np.zeros(len(activities), dtype=np.float64)
        for index in reached:
            gap = np.maximum(targets[index] - activities[:, index], 0.0)
            passed &= gap == 0
            gaps += gap
        if passed.all():
            return utilities
        all_activities = self._activities_array()
        all_utilities = self._scalarize(all_activities)
        floor = float(all_utilities.min()) - max(
            1.0, float(np.ptp(all_utilities)) if len(all_utilities) else 1.0
        )
        gated = np.asarray(utilities, dtype=np.float64).copy()
        gated[~passed] = floor - gaps[~passed]
        return gated

    def scores_for(self, sequences) -> np.ndarray:
        """Re-evaluate cached activities under the currently reached targets."""
        activities = self.activities_for(sequences)
        return self._apply_target_gate(activities, self._scalarize(activities))

    def uncertainties_for(self, sequences) -> np.ndarray | None:
        """Re-evaluate cached raw uncertainties for the current objective state."""
        values = np.asarray(sequences, dtype=str).reshape(-1)
        if not all(str(value) in self._activity_uncertainties for value in values):
            return None
        activities = self.activities_for(values)
        uncertainties = np.stack(
            [self._activity_uncertainties[str(value)] for value in values]
        )
        if isinstance(self.objective, MultiTaskObjective):
            return self.objective.uncertainty(
                activities,
                uncertainties,
                self.task_names,
                reached_targets=self._reached_targets,
            )
        coefficients = self.objective.coefficients(activities.shape[1], self.task_names)
        return np.sqrt((uncertainties**2) @ (coefficients**2))

    def thresholds_passed(self, sequences) -> np.ndarray:
        """Return whether sequences satisfy every target activated so far."""
        values = np.asarray(sequences, dtype=str).reshape(-1)
        if not self._reached_targets:
            return np.ones(len(values), dtype=bool)
        passed = np.ones(len(values), dtype=bool)
        present = np.asarray(
            [str(value) in self._activities for value in values], dtype=bool
        )
        if present.any():
            activities = self.activities_for(values[present])
            targets = self._targets(activities.shape[1])
            passed[present] = np.asarray(
                [
                    all(row[index] >= targets[index] for index in self._reached_targets)
                    for row in activities
                ],
                dtype=bool,
            )
        return passed

    @property
    def target_stop_reached(self) -> bool:
        """Whether a target-only objective has reached all thresholds jointly."""
        if not isinstance(self.objective, MultiTaskObjective):
            return False
        targets = self.objective.target_thresholds()
        if not targets or self.objective.has_open_objectives() or not self._activities:
            return False
        activities = self._activities_array()
        return bool(
            np.any(
                np.logical_and.reduce(
                    [
                        activities[:, index] >= target
                        for index, target in targets.items()
                    ]
                )
            )
        )

    def score(self, sequences) -> ScoreBatch:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        result = as_score_batch(self.scorer.score(values))
        activities = _activity_matrix(result.scores, len(values))
        if self.task_names is not None and len(self.task_names) != activities.shape[1]:
            raise ValueError("task_names must match the number of model outputs")
        for sequence, activity in zip(values, activities):
            self._activities[str(sequence)] = activity.copy()
        self._update_targets(activities)
        utilities = self.scores_for(values)

        utility_uncertainties = None
        if result.uncertainties is not None:
            uncertainty = _activity_matrix(result.uncertainties, len(values))
            if uncertainty.shape != activities.shape:
                raise ValueError("activity uncertainties must match activity outputs")
            for sequence, values_uncertainty in zip(values, uncertainty):
                self._activity_uncertainties[str(sequence)] = values_uncertainty.copy()
            utility_uncertainties = self.uncertainties_for(values)

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
            utilities = self.objective.scalarize_soft(
                activities,
                self.task_names,
                reached_targets=self._reached_targets,
            )
            uncertainty = None
            if uncertainties is not None:
                if uncertainties.shape != activities.shape:
                    raise ValueError(
                        "activity uncertainties must match activity outputs"
                    )
                ids, signs, center, scale, _ = self.objective._resolve(
                    activities.shape[1], self.task_names
                )
                active = self.objective._active_positions(ids, self._reached_targets)
                if not len(active):
                    return utilities, torch.zeros_like(utilities)
                ids = np.asarray(ids)[active].tolist()
                signs = signs[active]

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
                    weights = torch.full_like(margins, 1 / len(active))
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
