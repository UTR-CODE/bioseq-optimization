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
    directions: Sequence[str] | None = None
    weights: Sequence[float] | None = None
    stop_values: Sequence[float | tuple[float | None, float | None] | None] | None = None
    stop_tolerance: float | None = 0.5

    def __post_init__(self):
        if self.stop_tolerance is not None and (
            isinstance(self.stop_tolerance, bool)
            or not isinstance(self.stop_tolerance, Real)
            or not np.isfinite(self.stop_tolerance)
            or self.stop_tolerance < 0
        ):
            raise ValueError("stop_tolerance must be finite and non-negative, or None")
        if self.directions is not None:
            if isinstance(self.directions, str) or not self.directions:
                raise ValueError(
                    "directions must be a nonempty list of "
                    "maximize, minimize, or ignore"
                )
            directions = []
            for value in self.directions:
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if normalized not in {"maximize", "minimize", "ignore"}:
                        raise ValueError(
                            "directions entries must be maximize, minimize, or ignore"
                        )
                    directions.append(normalized)
                else:
                    raise ValueError(
                        "directions entries must be maximize, minimize, or ignore"
                    )
            if all(value == "ignore" for value in directions):
                raise ValueError("at least one output must be optimized")
            if len(self.high) or len(self.low):
                raise ValueError("use directions or high/low, not both")
            object.__setattr__(self, "directions", tuple(directions))
            stops = (
                [None] * len(directions)
                if self.stop_values is None
                else list(self.stop_values)
            )
            if len(stops) != len(directions):
                raise ValueError("stop_values must match directions length")
            for direction, value in zip(directions, stops):
                bounds = value if isinstance(value, (list, tuple)) else (value,)
                if len(bounds) not in (1, 2) or any(
                    bound is not None and (
                        isinstance(bound, bool)
                        or not isinstance(bound, Real)
                        or not np.isfinite(bound)
                    ) for bound in bounds
                ) or (isinstance(value, (list, tuple)) and len(bounds) != 2):
                    raise ValueError(
                        "stop_values entries must be finite numbers, (lower, upper), or None"
                    )
                if len(bounds) == 2 and all(b is not None for b in bounds) and bounds[0] > bounds[1]:
                    raise ValueError("stop_values lower bound must not exceed upper bound")
                if direction == "ignore" and value is not None:
                    raise ValueError("ignored outputs must have stop_values=None")
            if self.stop_values is not None:
                object.__setattr__(self, "stop_values", tuple(
                    tuple(v) if isinstance(v, (list, tuple)) else v for v in stops
                ))
        elif self.stop_values is not None:
            raise ValueError("stop_values requires directions")
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
            high = [i for i, value in enumerate(self.directions) if value == "maximize"]
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
        direction_targets = self.prediction_bounds(n_tasks)
        if direction_targets and bounded:
            raise ValueError(
                "stop_values cannot be combined with lower_bounds or upper_bounds"
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
        signs = np.asarray([1.0] * len(high) + [-1.0] * len(low))
        return selected, signs, center, scale, bounded

    def target_thresholds(self, n_tasks: int | None = None) -> dict[int, float]:
        """Return scalar directional bounds (not search termination targets)."""
        if self.directions is None:
            return {}
        if n_tasks is not None and len(self.directions) != n_tasks:
            raise ValueError(
                "directions must contain exactly one entry per model output"
            )
        return {
            index: float(value)
            for index, value in enumerate(self.stop_values or ())
            if value is not None and not isinstance(value, (list, tuple))
        }

    def prediction_bounds(self, n_tasks=None):
        """Inclusive raw-output bounds; scalar caps follow objective direction."""
        if n_tasks is not None and self.directions is not None and len(self.directions) != n_tasks:
            raise ValueError("directions must contain exactly one entry per model output")
        bounds = {}
        for index, value in enumerate(self.stop_values or ()):
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                lower, upper = value
            elif self.directions[index] == "maximize":
                lower, upper = None, value
            else:
                lower, upper = value, None
            if lower is not None or upper is not None:
                bounds[index] = (lower, upper)
        return bounds

    def feasible(self, activities):
        values = np.asarray(activities, dtype=float)
        passed = np.isfinite(values).all(axis=1)
        for index, (lower, upper) in self.prediction_bounds(values.shape[1]).items():
            if lower is not None:
                passed &= values[:, index] >= lower
            if upper is not None:
                passed &= values[:, index] <= upper
        return passed

    def maintenance_targets(self, n_tasks=None):
        """Inner edges of the one-sided maintenance bands in raw model units.

        No global reached state: each row stops earning reward independently,
        and immediately earns a deficit again on falling back outside its band.
        None explicitly opts into hard-bounds-only behavior.
        """
        if self.stop_tolerance is None:
            return {}
        targets = {}
        for i, (lower, upper) in self.prediction_bounds(n_tasks).items():
            if self.directions[i] == "maximize" and upper is not None:
                targets[i] = max(upper - self.stop_tolerance, lower) if lower is not None else upper - self.stop_tolerance
            elif self.directions[i] == "minimize" and lower is not None:
                targets[i] = min(lower + self.stop_tolerance, upper) if upper is not None else lower + self.stop_tolerance
        return targets

    def _maintenance_margins(self, activities, task_names=None):
        margins = self.margins(activities, task_names)
        ids, signs, _, scales, _ = self._resolve(activities.shape[1], task_names)
        for i, edge in self.maintenance_targets(activities.shape[1]).items():
            pos = ids.index(i)
            margins[:, pos] = np.minimum((activities[:, i] - edge) * signs[pos] / scales[i], 0.0)
        return margins

    def target_sign(self, index):
        return -1.0 if self.directions[index] == "minimize" else 1.0

    def has_open_objectives(self) -> bool:
        return bool(
            self.directions
            and any(
                value != "ignore" and stop is None
                for value, stop in zip(
                    self.directions,
                    self.stop_values or (None,) * len(self.directions),
                )
            )
        )

    def _active_positions(
        self,
        ids: Sequence[int],
        reached_targets: set[int] | frozenset[int] | None,
    ) -> np.ndarray:
        # Bounds never deactivate a task, even after a candidate touches a line.
        return np.arange(len(ids), dtype=int)

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
        activities = np.asarray(activities, dtype=float)
        margins = self._maintenance_margins(activities, task_names)
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
        pooled = shifted.mean(axis=1) if weights is None else shifted @ weights
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
        targets = self.maintenance_targets(activities.shape[1])
        if targets:
            margins = torch.stack([
                torch.minimum((activities[:, i] - targets[i]) * signs[pos] / scale[i],
                              torch.zeros_like(activities[:, i]))
                if i in targets else margins[:, pos]
                for pos, i in enumerate(ids)
            ], dim=1)
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
        activities = np.asarray(activities, dtype=float)
        margins = self._maintenance_margins(activities, task_names)
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
        aggregation_weights = np.array(aggregation_weights, copy=True)
        for i, edge in self.maintenance_targets(activities.shape[1]).items():
            pos = ids.index(i)
            ready = (activities[:, i] - edge) * self.target_sign(i) >= 0
            aggregation_weights[ready, pos] = 0.0
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
            "stop_values": list(self.stop_values)
            if self.stop_values is not None
            else [None] * n_tasks,
            "centers_or_bounds": center.tolist(),
            "scales": scale.tolist(),
            "thresholded": bounded,
            "stop_values_semantics": "inclusive_prediction_bounds",
            "stop_tolerance": self.stop_tolerance,
            "maintenance_targets": self.maintenance_targets(n_tasks),
            "maintenance_semantics": "per_candidate_capped_deficit_with_reversible_penalty",
            "prediction_bounds": {
                (task_names[i] if task_names else f"task_{i}"): list(bounds)
                for i, bounds in self.prediction_bounds(n_tasks).items()
            },
            "temperature": self.temperature,
            "uncertainty_method": "independent_first_order",
            "target_thresholds": {
                (task_names[index] if task_names else f"task_{index}"): value
                for index, value in self.target_thresholds(n_tasks).items()
            },
        }

    def evaluate(self, activities, task_names=None):
        ids, _, _, _, bounded = self._resolve(activities.shape[1], task_names)
        bounds = self.prediction_bounds(activities.shape[1])
        if bounds:
            ids = list(bounds)
            targets = self.maintenance_targets(activities.shape[1])
            ready = np.stack([
                (activities[:, i] - edge) * self.target_sign(i) >= 0
                for i, edge in targets.items()
            ], axis=1) if targets else None
            margins = np.stack([
                np.minimum(
                    activities[:, i] - lower if lower is not None else np.full(len(activities), np.inf),
                    upper - activities[:, i] if upper is not None else np.full(len(activities), np.inf),
                ) for i, (lower, upper) in bounds.items()
            ], axis=1)
            return {
                "task_names": [task_names[i] if task_names else f"task_{i}" for i in ids],
                "margins": margins.tolist(),
                "worst_margin": margins.min(axis=1).tolist(),
                "satisfied": (margins >= 0).tolist(),
                "all_satisfied": self.feasible(activities).tolist(),
                "maintenance_task_indices": list(targets),
                "maintenance_satisfied": None if ready is None else ready.tolist(),
                "all_maintenance_satisfied": None if ready is None else (
                    ready.all(axis=1) & self.feasible(activities)
                ).tolist(),
            }
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


class CustomObjective:
    """A tensor function of named model outputs, returning one score per sequence."""

    def __init__(self, function):
        self.function = function

    def describe(self, n_tasks, task_names=None):
        return {"type": "custom", "function": getattr(self.function, "__name__", type(self.function).__name__)}

    def scalarize(self, activities, task_names=None):
        import torch

        with torch.no_grad():
            values = self.scalarize_soft(torch.as_tensor(activities), task_names).numpy()
        if not np.isfinite(values).all():
            raise ValueError("The custom objective returned a non-finite score")
        return values

    def scalarize_soft(self, activities, task_names=None):
        import torch

        names = task_names or tuple(f"task_{i}" for i in range(activities.shape[1]))
        scores = self.function(dict(zip(names, activities.unbind(dim=1))))
        if not isinstance(scores, torch.Tensor) or scores.shape != (len(activities),):
            raise ValueError("A custom objective must return a PyTorch tensor with one score per sequence")
        return scores


class ObjectiveScorer:
    """Turn raw model activities into scalar utilities and retain activities."""

    def __init__(
        self,
        scorer: SequenceScorer,
        objective: TargetObjective | MultiTaskObjective | list[str] | tuple[str, ...],
        weights: Sequence[float] | None = None,
        stop_values: Sequence[float | tuple[float | None, float | None] | None] | None = None,
        stop_tolerance: float | None = 0.5,
    ) -> None:
        self.scorer = scorer
        if isinstance(objective, (list, tuple)):
            self.objective = MultiTaskObjective(
                directions=objective,
                aggregation="mean",
                weights=weights,
                stop_values=stop_values,
                stop_tolerance=stop_tolerance,
            )
        else:
            if stop_values is not None:
                if not isinstance(objective, MultiTaskObjective):
                    raise ValueError("stop_values requires a directions objective")
                objective = replace(objective, stop_values=stop_values, stop_tolerance=stop_tolerance)
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
        # Kept for compatibility: prediction bounds are stationary and never
        # deactivate objectives or latch state across unrelated candidates.
        return

    @property
    def has_prediction_bounds(self) -> bool:
        return isinstance(self.objective, MultiTaskObjective) and bool(
            self.objective.prediction_bounds()
        )

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
        if not self.has_prediction_bounds:
            return utilities
        passed = self.objective.feasible(activities)
        gated = np.asarray(utilities, dtype=np.float64).copy()
        # A finite sentinel is useful for diagnostics, but never substitutes
        # for the explicit feasibility mask used by candidate selection.
        gated[~passed] = -1e30
        return gated

    def scores_for(self, sequences) -> np.ndarray:
        """Score cached activities with stationary objectives and hard bounds."""
        activities = self.activities_for(sequences)
        return self._apply_target_gate(activities, self._scalarize(activities))

    def uncertainties_for(self, sequences) -> np.ndarray | None:
        """Re-evaluate cached raw uncertainties for the current objective state."""
        if isinstance(self.objective, CustomObjective):
            return None
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
        """Return feasibility from the first round; unscored rows fail closed."""
        values = np.asarray(sequences, dtype=str).reshape(-1)
        if not self.has_prediction_bounds:
            return np.ones(len(values), dtype=bool)
        passed = np.zeros(len(values), dtype=bool)
        present = np.asarray(
            [str(value) in self._activities for value in values], dtype=bool
        )
        if present.any():
            activities = self.activities_for(values[present])
            passed[present] = self.objective.feasible(activities)
        return passed

    @property
    def target_stop_reached(self) -> bool:
        """Prediction bounds do not terminate search; budget/plateau still do."""
        return False

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
            if self.has_prediction_bounds:
                # Differentiable guidance only. Hard decoded candidates are
                # separately scored and rejected before selection/training.
                # Reflect the utility outside the box so overshoot is not
                # rewarded even for a soft PWM whose prediction differs from
                # its argmax sequence.
                violation = torch.zeros_like(utilities)
                _, _, _, scales, _ = self.objective._resolve(
                    activities.shape[1], self.task_names
                )
                for index, (lower, upper) in self.objective.prediction_bounds().items():
                    if lower is not None:
                        violation = violation + torch.relu(lower - activities[:, index]) / scales[index]
                    if upper is not None:
                        violation = violation + torch.relu(activities[:, index] - upper) / scales[index]
                utilities = utilities - 2.0 * violation
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
                targets = self.objective.maintenance_targets(activities.shape[1])
                if targets:
                    margins = torch.stack([
                        torch.minimum((activities[:, i] - targets[i]) * signs[pos] / scale[i],
                                      torch.zeros_like(activities[:, i]))
                        if i in targets else margins[:, pos]
                        for pos, i in enumerate(ids)
                    ], dim=1)
                if self.objective.aggregation == "mean":
                    row = self.objective._active_weights(ids, np.arange(len(ids)), activities.shape[1])
                    weights = torch.full_like(margins, 1 / len(active)) if row is None else tensor(row).expand_as(margins)
                elif self.objective.aggregation == "worst":
                    weights = (margins == margins.amin(dim=1, keepdim=True)).to(
                        activities.dtype
                    )
                    weights = weights / weights.sum(dim=1, keepdim=True)
                else:
                    row = self.objective._active_weights(ids, np.arange(len(ids)), activities.shape[1])
                    weights = torch.softmax(
                        -margins / self.objective.temperature + (0 if row is None else torch.log(tensor(row).clamp_min(1e-12))), dim=1
                    )
                mask = torch.stack([
                    ((activities[:, i] - targets[i]) * signs[pos] < 0).to(activities.dtype)
                    if i in targets else torch.ones_like(activities[:, i])
                    for pos, i in enumerate(ids)
                ], dim=1)
                weights = weights * mask
                uncertainty = torch.linalg.vector_norm(
                    uncertainties[:, ids] / tensor(scale[ids]) * weights, dim=1
                )
            return utilities, uncertainty
        if isinstance(self.objective, CustomObjective):
            return self.objective.scalarize_soft(activities, self.task_names), None
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
