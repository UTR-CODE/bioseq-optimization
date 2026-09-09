"""Budget-free optimization with plateau detection and q-fraction selections."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .algorithms import _REGISTRY, _initial, available_algorithms
from .api import (
    CachedScorer,
    OptimizationConfig,
    OptimizationResult,
    ScoreBatch,
    SequenceScorer,
    SequenceSpace,
)
from .objectives import ObjectiveScorer, TargetObjective


class _SaturationReached(RuntimeError):
    pass


@dataclass(frozen=True)
class SaturationConfig:
    """Rules for stopping after the top-candidate archive reaches a plateau.

    Query limits default to multiples of ``num_candidates`` so a user can run
    without choosing an arbitrary optimization budget. ``maximum_queries`` is
    still retained as a safety ceiling.
    """

    fractions: tuple[float, ...] = (0.8, 0.9)
    minimum_queries: int | None = None
    plateau_patience_queries: int | None = None
    maximum_queries: int | None = None
    material_improvement: float = 1e-4

    def resolve(self, num_candidates: int) -> tuple[int, int, int]:
        fractions = tuple(float(value) for value in self.fractions)
        if not fractions or any(not 0 < value <= 1 for value in fractions):
            raise ValueError("saturation fractions must lie in (0, 1]")
        if len(set(fractions)) != len(fractions):
            raise ValueError("saturation fractions must be unique")
        if not np.isfinite(self.material_improvement) or self.material_improvement <= 0:
            raise ValueError("material_improvement must be finite and positive")
        minimum = self.minimum_queries or 10 * num_candidates
        patience = self.plateau_patience_queries or 20 * num_candidates
        maximum = self.maximum_queries or 1000 * num_candidates
        if minimum < num_candidates or patience < 1 or maximum < minimum:
            raise ValueError("invalid saturation query limits")
        return int(minimum), int(patience), int(maximum)


@dataclass
class SaturationResult:
    """Selections at requested fractions of the final observed plateau gain."""

    selections: dict[float, OptimizationResult]
    total_query_count: int
    knee_query_count: int
    stop_reason: str
    trajectory: list[dict[str, float | int]]

    def at(self, fraction: float) -> OptimizationResult:
        key = float(fraction)
        if key not in self.selections:
            raise KeyError(
                f"fraction {fraction} was not requested; choose from {tuple(self.selections)}"
            )
        return self.selections[key]

    def __getitem__(self, fraction: float) -> OptimizationResult:
        return self.at(fraction)


class _SaturationMonitor:
    def __init__(
        self,
        num_candidates: int,
        minimum_queries: int,
        patience_queries: int,
        material_improvement: float,
    ) -> None:
        self.num_candidates = num_candidates
        self.minimum_queries = minimum_queries
        self.patience_queries = patience_queries
        self.material_improvement = material_improvement
        self.archive: dict[str, tuple[float, float]] = {}
        self.snapshots: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        self.material_level = -np.inf
        self.last_material_query = 0

    def observe(
        self, sequences: np.ndarray, batch: ScoreBatch, query_count: int
    ) -> None:
        uncertainty = (
            np.zeros_like(batch.scores)
            if batch.uncertainties is None
            else batch.uncertainties
        )
        for sequence, score, std in zip(sequences, batch.scores, uncertainty):
            key = str(sequence)
            value = (float(score), float(std))
            if key not in self.archive or value[0] > self.archive[key][0]:
                self.archive[key] = value
        if len(self.archive) < self.num_candidates:
            return
        ranked = sorted(self.archive.items(), key=lambda item: (-item[1][0], item[0]))[
            : self.num_candidates
        ]
        selected_sequences = np.asarray([item[0] for item in ranked], dtype=str)
        selected_scores = np.asarray([item[1][0] for item in ranked], dtype=np.float64)
        selected_stds = np.asarray([item[1][1] for item in ranked], dtype=np.float64)
        if self.snapshots and query_count == self.snapshots[-1][0]:
            self.snapshots[-1] = (
                query_count,
                selected_sequences,
                selected_scores,
                selected_stds,
            )
        else:
            self.snapshots.append(
                (query_count, selected_sequences, selected_scores, selected_stds)
            )
        top_mean = float(selected_scores.mean())
        if top_mean > self.material_level + self.material_improvement:
            self.material_level = top_mean
            self.last_material_query = query_count
        if (
            query_count >= self.minimum_queries
            and query_count - self.last_material_query >= self.patience_queries
        ):
            raise _SaturationReached


class _TrackingScorer(CachedScorer):
    def __init__(self, *args, monitor: _SaturationMonitor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.monitor = monitor

    def score(self, sequences) -> ScoreBatch:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        batch = super().score(values)
        self.monitor.observe(values, batch, self.query_count)
        return batch


def _selection_result(
    algorithm: str,
    snapshot: tuple[int, np.ndarray, np.ndarray, np.ndarray],
    *,
    fraction: float,
    total_queries: int,
    knee_queries: int,
    baseline: float,
    plateau: float,
    objective_scorer: ObjectiveScorer,
    trajectory: list[dict[str, float | int]],
    stop_reason: str,
) -> OptimizationResult:
    query_count, sequences, scores, uncertainties = snapshot
    result = OptimizationResult(
        algorithm=algorithm,
        sequences=sequences,
        scores=scores,
        uncertainties=uncertainties,
        query_count=query_count,
        metadata={
            "selection": "fraction_of_observed_plateau_gain",
            "saturation_fraction": fraction,
            "baseline_top_mean": baseline,
            "plateau_top_mean": plateau,
            "total_convergence_queries": total_queries,
            "knee_query_count": knee_queries,
            "stop_reason": stop_reason,
        },
        trajectory=trajectory,
    )
    activities = objective_scorer.activities_for(sequences)
    result.attach_activities(
        activities,
        objective_scorer.resolved_task_names(activities.shape[1]),
    )
    result.metadata["objective"] = objective_scorer.objective.describe(
        activities.shape[1], objective_scorer.task_names
    )
    return result


def optimize_to_saturation(
    scorer: SequenceScorer,
    space: SequenceSpace,
    *,
    algorithm: str,
    num_candidates: int = 100,
    initial=None,
    objective: TargetObjective | None = None,
    saturation: SaturationConfig | None = None,
    seed: int = 0,
    count_repeated: bool = False,
    **optimizer_parameters: Any,
) -> SaturationResult:
    """Run until a plateau and return q-fraction candidate sets.

    A q80 result is the earliest archive whose top-candidate mean reaches 80%
    of the gain between the first valid archive and the final observed plateau.
    """
    key = str(algorithm).strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown algorithm {algorithm!r}; choose from {', '.join(available_algorithms())}"
        )
    options = saturation or SaturationConfig()
    minimum, patience, maximum = options.resolve(num_candidates)
    config = OptimizationConfig(
        budget=maximum,
        num_candidates=num_candidates,
        **optimizer_parameters,
    )
    rng = np.random.default_rng(seed)
    random.seed(seed)
    prepared = _initial(initial, space, config, rng)
    objective_scorer = ObjectiveScorer(scorer, objective or TargetObjective())
    monitor = _SaturationMonitor(
        num_candidates,
        minimum,
        patience,
        options.material_improvement,
    )
    proposal_count_algorithms = {
        "ledidi",
        "cbas",
        "vanilla_activation_maximization",
        "fast_seqprop",
        "den",
    }
    budgeted = _TrackingScorer(
        objective_scorer,
        maximum,
        count_repeated=count_repeated or key in proposal_count_algorithms,
        monitor=monitor,
    )
    stop_reason = "algorithm_stopped"
    try:
        _REGISTRY[key](prepared, budgeted, space, config, rng)
        if budgeted.query_count >= maximum:
            stop_reason = "maximum_queries"
    except _SaturationReached:
        stop_reason = "plateau"

    if not monitor.snapshots:
        raise RuntimeError(
            "optimizer did not produce enough candidates for saturation analysis"
        )
    query_counts = np.asarray([item[0] for item in monitor.snapshots], dtype=np.float64)
    top_means = np.asarray(
        [item[2].mean() for item in monitor.snapshots], dtype=np.float64
    )
    baseline, plateau = float(top_means[0]), float(top_means[-1])
    if len(top_means) == 1 or plateau - baseline <= options.material_improvement:
        knee_index = 0
    else:
        x_normalized = (query_counts - query_counts[0]) / max(
            float(query_counts[-1] - query_counts[0]), 1.0
        )
        y_normalized = (top_means - baseline) / max(plateau - baseline, 1e-12)
        knee_index = int(np.argmax(y_normalized - x_normalized))
    knee_queries = int(query_counts[knee_index])
    trajectory = [
        {
            "query_count": int(query_count),
            "top_mean_score": float(top_mean),
            "is_knee": int(index == knee_index),
        }
        for index, (query_count, top_mean) in enumerate(zip(query_counts, top_means))
    ]
    selections: dict[float, OptimizationResult] = {}
    for fraction in options.fractions:
        target = baseline + float(fraction) * (plateau - baseline)
        index = int(np.flatnonzero(top_means >= target - 1e-12)[0])
        selections[float(fraction)] = _selection_result(
            key,
            monitor.snapshots[index],
            fraction=float(fraction),
            total_queries=budgeted.query_count,
            knee_queries=knee_queries,
            baseline=baseline,
            plateau=plateau,
            objective_scorer=objective_scorer,
            trajectory=trajectory,
            stop_reason=stop_reason,
        )
    return SaturationResult(
        selections=selections,
        total_query_count=budgeted.query_count,
        knee_query_count=knee_queries,
        stop_reason=stop_reason,
        trajectory=trajectory,
    )
