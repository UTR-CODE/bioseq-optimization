"""Plateau surveillance and score-threshold or gain-stage selection."""

from __future__ import annotations

import random
from heapq import merge, nsmallest
from itertools import islice
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .api import (
    CachedScorer,
    OptimizationConfig,
    OptimizationResult,
    ScoreBatch,
    SequenceScorer,
)
from .constraints import ConstraintSpec
from .objectives import MultiTaskObjective, ObjectiveScorer, TargetObjective


class _SaturationReached(RuntimeError):
    pass


class _ScoreThresholdReached(RuntimeError):
    pass


@dataclass(frozen=True)
class SaturationConfig:
    """Rules for stopping after the top-candidate archive reaches a plateau.

    Public optimization uses plateau-based stopping without a maximum query
    ceiling. The public default returns the 90% observed-gain archive.
    Unspecified minimum/patience windows use 10/40 times the larger of the
    initial search population and the requested retained candidate count.
    The optimizer also floors unspecified patience at three nominal proposal
    batches, to accommodate large neighborhood expansions.
    Candidate archives respect the requested size and diversity. Historical
    elbow diagnostics are kept for reproducibility of internal analyses.
    """

    fractions: tuple[float, ...] = (0.9,)
    minimum_queries: int | None = None
    plateau_patience_queries: int | None = None
    plateau_patience_rounds: int | None = 20
    maximum_queries: int | None = None
    material_improvement: float = 1e-2
    relative_material_improvement: float = 1e-3
    selection: str = "fraction"
    knee_fraction: float = 0.9
    knee_fraction_tolerance: float = 0.1
    diversity_pool_size: int = 20_000
    relaxed_knee: bool = True
    knee_minimum_gain_fraction: float = 0.85
    selection_minimum_gain_fraction: float = 0.8

    def resolve(
        self, num_candidates: int, population_size: int | None = None
    ) -> tuple[int, int, int | None]:
        fractions = tuple(float(value) for value in self.fractions)
        if self.plateau_patience_rounds is not None and (
            isinstance(self.plateau_patience_rounds, bool)
            or not isinstance(self.plateau_patience_rounds, (int, np.integer))
            or self.plateau_patience_rounds < 10
        ):
            raise ValueError("plateau_patience_rounds must be at least 10, or None for query-based patience")
        if not isinstance(self.relaxed_knee, bool):
            raise ValueError("relaxed_knee must be boolean")
        if self.selection not in {"knee_fraction", "knee", "fraction"}:
            raise ValueError("selection must be 'knee_fraction', 'knee' or 'fraction'")
        if not 0 < self.knee_fraction <= 1:
            raise ValueError("knee_fraction must lie in (0, 1]")
        if not 0 <= self.selection_minimum_gain_fraction <= 1:
            raise ValueError(
                "selection_minimum_gain_fraction must lie in [0, 1]"
            )
        if not 0 < self.knee_minimum_gain_fraction <= 1:
            raise ValueError("knee_minimum_gain_fraction must lie in (0, 1]")
        if (
            self.selection_minimum_gain_fraction
            > self.knee_minimum_gain_fraction
        ):
            raise ValueError(
                "selection_minimum_gain_fraction cannot exceed "
                "knee_minimum_gain_fraction"
            )
        if (
            not np.isfinite(self.knee_fraction_tolerance)
            or self.knee_fraction_tolerance < 0
        ):
            raise ValueError("knee_fraction_tolerance must be finite and non-negative")
        if (
            not isinstance(self.diversity_pool_size, (int, np.integer))
            or isinstance(self.diversity_pool_size, (bool, np.bool_))
            or self.diversity_pool_size < num_candidates
        ):
            raise ValueError(
                "diversity_pool_size must be an integer at least num_candidates"
            )
        if not fractions or any(not 0 < value <= 1 for value in fractions):
            raise ValueError("saturation fractions must lie in (0, 1]")
        if len(set(fractions)) != len(fractions):
            raise ValueError("saturation fractions must be unique")
        if (
            not np.isfinite(self.material_improvement)
            or self.material_improvement < 1e-2
        ):
            raise ValueError("material_improvement must be finite and at least 1e-2")
        if (
            not np.isfinite(self.relative_material_improvement)
            or self.relative_material_improvement < 0
        ):
            raise ValueError(
                "relative_material_improvement must be finite and non-negative"
            )
        # Search population and retained output size need not be equal. In
        # particular, one gradient update proposes the entire initial population.
        scale = max(num_candidates, population_size or num_candidates)
        minimum = 10 * scale if self.minimum_queries is None else self.minimum_queries
        patience = (
            40 * scale
            if self.plateau_patience_queries is None
            else self.plateau_patience_queries
        )
        maximum = self.maximum_queries
        if (
            minimum < num_candidates
            or patience < 1
            or (maximum is not None and maximum < minimum)
        ):
            raise ValueError("invalid saturation query limits")
        return int(minimum), int(patience), None if maximum is None else int(maximum)


def resolve_search_config(saturation, gain_fraction, score_threshold, gain_fractions=None):
    """Normalize public plateau/gain and threshold search configuration."""
    modes = (saturation, gain_fraction, gain_fractions, score_threshold)
    if sum(value is not None for value in modes) > 1:
        raise ValueError("Choose one of saturation, gain_fraction, gain_fractions, or score_threshold")
    if saturation is not None:
        if not isinstance(saturation, SaturationConfig):
            raise TypeError("saturation must be a SaturationConfig")
        if saturation.maximum_queries is not None or saturation.selection != "fraction":
            raise ValueError("Only plateau gain selection is supported in saturation; omit maximum_queries and use selection='fraction'")
    if score_threshold is not None and (
        isinstance(score_threshold, (bool, np.bool_)) or not np.isfinite(score_threshold)
    ):
        raise ValueError("score_threshold must be finite")
    if saturation is not None:
        values = saturation.fractions
    elif gain_fractions is not None:
        values = gain_fractions
    else:
        values = (0.9 if gain_fraction is None else gain_fraction,)
    if isinstance(values, (str, bytes)):
        raise TypeError("gain_fractions must be a sequence of numbers")
    values = tuple(values)
    fractions = tuple(float(value) for value in values)
    if not fractions or any(isinstance(value, (bool, np.bool_)) for value in values) or any(
        not 0 < fraction <= 1 for fraction in fractions
    ):
        raise ValueError("Specify one or more gain fractions in (0, 1]")
    if len(set(fractions)) != len(fractions):
        raise ValueError("Gain fractions must be distinct")
    return replace(saturation, fractions=fractions) if saturation is not None else SaturationConfig(fractions=fractions)


@dataclass
class SaturationResult:
    """Selected elbow-relative archive plus elbow and legacy final-gain archives."""

    selections: dict[float, OptimizationResult]
    total_query_count: int
    knee_query_count: int
    stop_reason: str
    trajectory: list[dict[str, float | int]]
    knee: OptimizationResult | None = None
    selection: str = "knee_fraction"
    knee_fraction_result: OptimizationResult | None = None
    threshold_result: OptimizationResult | None = None

    @property
    def selected(self) -> OptimizationResult:
        if self.threshold_result is not None:
            return self.threshold_result
        if self.selection == "knee_fraction" and self.knee_fraction_result is not None:
            return self.knee_fraction_result
        if self.selection == "knee" and self.knee is not None:
            return self.knee
        return self.selections[max(self.selections)]

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
        diversity: int,
        relative_material_improvement: float = 1e-3,
        patience_rounds: int | None = None,
        score_threshold: float | None = None,
        report_progress=None,
    ) -> None:
        self.num_candidates = num_candidates
        self.minimum_queries = minimum_queries
        self.patience_queries = patience_queries
        self.material_improvement = max(1e-2, material_improvement)
        self.relative_material_improvement = relative_material_improvement
        self.patience_rounds = patience_rounds
        self.score_threshold = score_threshold
        self.report_progress = report_progress
        self.observation_round = 0
        self.last_material_round = 0
        self.diversity = diversity
        self.archive: dict[str, tuple[float, float]] = {}
        self.discovery_queries: dict[str, int] = {}
        self._prefix_size = max(128, 2 * num_candidates)
        self._ranking_limit = self._prefix_size
        self._ranking: list[tuple[float, str]] = []
        self._selection = None
        self._cutoff = None
        self.snapshots: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        self.material_level = -np.inf
        self.observed_peak = -np.inf
        self.last_material_threshold = material_improvement
        self.last_material_query = 0
        self.initial_mean = None
        self.initial_query = 0
        self._available_count = 0
        self._collection_level = -np.inf
        self._collection_peak = -np.inf
        self._collection_query = 0

    @property
    def ranking(self) -> list[tuple[float, str]]:
        """Return the complete ordering for diagnostics and compatibility."""
        return sorted((-value[0], key) for key, value in self.archive.items())

    def _ensure_ranking_prefix(self, limit: int) -> None:
        """Materialize only the top archive prefix needed by selection."""
        if limit <= self._ranking_limit and len(self._ranking) >= limit:
            return
        self._ranking = nsmallest(
            limit,
            ((-value[0], key) for key, value in self.archive.items()),
        )
        self._ranking_limit = limit

    def reset(self, query_count: int) -> None:
        """Restart plateau tracking after a stopping line changes the objective."""
        self.archive.clear()
        self.discovery_queries.clear()
        self._ranking.clear()
        self._ranking_limit = self._prefix_size
        self._selection = None
        self._cutoff = None
        self.snapshots.clear()
        self.material_level = -np.inf
        self.observed_peak = -np.inf
        self.last_material_threshold = self.material_improvement
        self.last_material_query = query_count
        self.initial_mean = None
        self.initial_query = query_count
        self._available_count = 0
        self._collection_level = -np.inf
        self._collection_peak = -np.inf
        self._collection_query = query_count
        self.observation_round = self.last_material_round = 0

    def _dynamic_material_threshold(self, current: float, reference: float) -> float:
        scale = max(
            1.0,
            abs(float(current)),
            abs(float(reference)) if np.isfinite(reference) else 0.0,
        )
        return max(
            self.material_improvement,
            self.relative_material_improvement * scale,
        )

    def observe(
        self, sequences: np.ndarray, batch: ScoreBatch, query_count: int,
        *, advance_round: bool = True,
    ) -> None:
        if advance_round:
            self.observation_round += 1
        uncertainty = (
            np.zeros_like(batch.scores)
            if batch.uncertainties is None
            else batch.uncertainties
        )
        if self.initial_mean is None and len(batch.scores):
            self.initial_mean = float(batch.scores.mean())
            self.initial_query = query_count
        # Consolidate a batch before touching the ordered archive.  Inserting
        # every candidate independently into a Python list shifts O(N) objects
        # per insertion and becomes quadratic for long searches.
        changes: dict[str, tuple[float, float]] = {}
        for sequence, score, std in zip(sequences, batch.scores, uncertainty):
            key = str(sequence)
            value = (float(score), float(std))
            previous = changes.get(key, self.archive.get(key))
            if previous is None or value[0] > previous[0]:
                changes[key] = value

        selected_keys = (
            {item[0] for item in self._selection}
            if self._selection is not None
            else set()
        )
        old_orders = {
            (-self.archive[key][0], key) for key in changes if key in self.archive
        }
        new_orders = []
        potentially_relevant = []
        for key, value in changes.items():
            order = (-value[0], key)
            if key not in self.archive:
                self.discovery_queries[key] = query_count
            self.archive[key] = value
            new_orders.append(order)
            if self._cutoff is None or order <= self._cutoff:
                potentially_relevant.append((key, order, key in selected_keys))

        if new_orders:
            unchanged = (
                (item for item in self._ranking if item not in old_orders)
                if old_orders
                else iter(self._ranking)
            )
            self._ranking = list(
                islice(
                    merge(unchanged, sorted(new_orders)),
                    self._ranking_limit,
                )
            )

        from .algorithms import _hamming_at_least, _select_diverse

        relevant_change = self._selection is None
        if not relevant_change and potentially_relevant:
            # Greedy diversity selection is unchanged when a newly inserted
            # candidate is rejected by an already-selected predecessor.  This
            # cheap incremental test avoids rescanning a multi-thousand-entry
            # prefix after most model batches.
            for key, order, was_selected in potentially_relevant:
                if was_selected or self.diversity == 0:
                    relevant_change = True
                    break
                can_enter = True
                for selected in self._selection:
                    selected_order = (-selected[1][0], selected[0])
                    if selected_order >= order:
                        break
                    if not _hamming_at_least(key, selected[0], self.diversity):
                        can_enter = False
                        break
                if can_enter:
                    relevant_change = True
                    break

        # A successful greedy prefix is exactly the full-archive result.
        # Never run alternate-start fallback until the full ranking is seen.
        limit = min(len(self.archive), self._prefix_size)
        self._ensure_ranking_prefix(limit)
        ranked = self._selection
        while relevant_change:
            prefix = [(key, self.archive[key]) for _, key in self._ranking[:limit]]
            complete = limit == len(self.archive)
            ranked = _select_diverse(
                prefix, self.num_candidates, self.diversity, fallback=False
            )
            if len(ranked) == self.num_candidates:
                self._selection = ranked
                self._cutoff = (-ranked[-1][1][0], ranked[-1][0])
                self._prefix_size = limit
                break
            if complete:
                ranked = _select_diverse(prefix, self.num_candidates, self.diversity)
                self._selection = None
                self._cutoff = None
                break
            limit = min(len(self.archive), 2 * limit)
            self._ensure_ranking_prefix(limit)
        if len(ranked) < self.num_candidates:
            # Collection phase: never declare plateau or return a short set.
            # A stagnant small search population must not loop forever either.
            level = (
                float(np.mean([value[0] for _, value in ranked])) if ranked else -np.inf
            )
            threshold = self._dynamic_material_threshold(level, self._collection_peak)
            if (
                len(ranked) > self._available_count
                or level > self._collection_peak + threshold
            ):
                self._available_count = len(ranked)
                self._collection_level = level
                self._collection_query = query_count
            self._collection_peak = max(self._collection_peak, level)
            if (
                query_count >= self.minimum_queries
                and query_count - self._collection_query >= self.patience_queries
            ):
                raise RuntimeError(
                    f"candidate collection stalled: only {len(ranked)} candidates satisfy "
                    f"diversity={self.diversity}; {self.num_candidates} required. "
                    "Increase initial population or patience, or lower diversity."
                )
            return
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
        if advance_round and self.report_progress is not None:
            self.report_progress(self.observation_round, top_mean)
        if self.score_threshold is not None and float(selected_scores.max()) >= self.score_threshold:
            raise _ScoreThresholdReached
        threshold = self._dynamic_material_threshold(top_mean, self.material_level)
        self.last_material_threshold = threshold
        if not np.isfinite(self.observed_peak):
            self.material_level = top_mean
            self.last_material_query = query_count
            self.last_material_round = self.observation_round
        elif top_mean - self.material_level > threshold:
            self.material_level = top_mean
            self.last_material_query = query_count
            self.last_material_round = self.observation_round
        self.observed_peak = max(self.observed_peak, top_mean)
        if advance_round and (
            query_count >= self.minimum_queries
            and (
                self.observation_round - self.last_material_round >= self.patience_rounds
                if self.patience_rounds is not None
                else query_count - self.last_material_query >= self.patience_queries
            )
        ):
            raise _SaturationReached


class _TrackingScorer(CachedScorer):
    def __init__(self, *args, monitor: _SaturationMonitor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self._monitor_objective_version = self.objective_version
        self._observation_group_depth = 0

    def begin_observation_group(self) -> None:
        """Group several dependent model batches into one optimizer round."""
        self._observation_group_depth += 1

    def end_observation_group(self) -> None:
        if self._observation_group_depth < 1:
            raise RuntimeError("observation group was not started")
        self._observation_group_depth -= 1
        if self._observation_group_depth == 0 and self.monitor.archive:
            self.monitor.observe(
                np.empty(0, dtype=str), ScoreBatch(np.empty(0), np.empty(0)),
                self.query_count, advance_round=True,
            )

    def score(self, sequences) -> ScoreBatch:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        batch = super().score(values)
        objective_changed = self.objective_version != self._monitor_objective_version
        if objective_changed:
            self.monitor.reset(self.query_count)
            self._monitor_objective_version = self.objective_version
        tracked = (
            np.asarray(list(self._cache), dtype=str) if objective_changed else values
        )
        passed = self.constraints_passed(tracked)
        if passed.any():
            accepted = tracked[passed]
            self.monitor.observe(
                accepted,
                ScoreBatch(
                    self.current_scores(accepted),
                    self.current_uncertainties(accepted),
                ),
                self.query_count,
                advance_round=self._observation_group_depth == 0,
            )
        elif self.monitor.archive:
            # Rejected proposals still consume queries and plateau patience.
            # Otherwise an all-infeasible generator can run forever when the
            # saturation configuration has no maximum query count.
            self.monitor.observe(
                np.empty(0, dtype=str), ScoreBatch(np.empty(0), np.empty(0)),
                self.query_count,
                advance_round=self._observation_group_depth == 0,
            )
        return batch


def _selection_result(
    algorithm: str,
    snapshot: tuple[int, np.ndarray, np.ndarray, np.ndarray],
    *,
    fraction: float | None,
    total_queries: int,
    knee_queries: int,
    baseline: float,
    plateau: float,
    objective_scorer: ObjectiveScorer,
    budgeted: CachedScorer,
    trajectory: list[dict[str, float | int]],
    stop_reason: str,
    diversity: int,
) -> OptimizationResult:
    query_count, sequences, scores, uncertainties = snapshot
    result = OptimizationResult(
        algorithm=algorithm,
        sequences=sequences,
        scores=scores,
        uncertainties=uncertainties,
        query_count=query_count,
        metadata={
            "selection": "knee"
            if fraction is None
            else "fraction_of_observed_plateau_gain",
            "saturation_fraction": fraction,
            "baseline_top_mean": baseline,
            "plateau_top_mean": plateau,
            "total_convergence_queries": total_queries,
            "knee_query_count": knee_queries,
            "stop_reason": stop_reason,
            "diversity": diversity,
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
    if isinstance(objective_scorer.objective, MultiTaskObjective):
        result.metadata["objective_evaluation"] = objective_scorer.objective.evaluate(
            activities, objective_scorer.task_names
        )
    if budgeted.constraint_evaluator is not None:
        evaluator = budgeted.constraint_evaluator
        result.attach_constraint_scores(
            evaluator.probabilities_for(sequences), evaluator.names
        )
        result.metadata["constraint"] = {
            **evaluator.describe(),
            "screened_query_count": int(evaluator.evaluated_counts[0]),
            "counted_query_count": total_queries,
            "model_query_count": budgeted.model_query_count,
        }
    return result


def _knee_index(queries, means, material_improvement):
    """Maximum normalized gain above the start/end chord; no inferred plateau.

    Flat curves use the initial archive. Without a positive interior bend,
    use the final archive instead of accidentally choosing the initial one.
    """
    if len(means) < 2 or np.max(means) - means[0] <= material_improvement:
        return 0
    if means[-1] <= means[0] + material_improvement:
        return int(np.argmax(means))
    x = (queries - queries[0]) / max(float(queries[-1] - queries[0]), 1.0)
    y = (means - means[0]) / (means[-1] - means[0])
    distance = y - x
    return int(np.argmax(distance)) if np.max(distance) > 1e-12 else len(means) - 1


def _first_eligible_index(means, target):
    hits = np.flatnonzero(means >= target - 1e-12)
    return int(hits[0]) if len(hits) else int(np.argmax(means))


def _pick_tied_candidate(
    indices: np.ndarray,
    primary: np.ndarray,
    deviations: np.ndarray,
    scores: np.ndarray,
    sequences: np.ndarray,
) -> int:
    """Pick a deterministic maximum-primary candidate near the quality target."""
    best = float(np.max(primary[indices]))
    tied = indices[primary[indices] == best]
    nearest = float(np.min(deviations[tied]))
    tied = tied[deviations[tied] == nearest]
    highest = float(np.max(scores[tied]))
    tied = tied[scores[tied] == highest]
    if len(tied) == 1:
        return int(tied[0])
    return int(min(tied, key=lambda index: str(sequences[int(index)])))


def _novelty_indices(
    sequences: np.ndarray,
    scores: np.ndarray,
    target: float,
    count: int,
    minimum_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily maximize the product of distances to the selected set."""
    size = len(sequences)
    if not size or count < 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    deviations = np.abs(scores - target)
    if count == 1:
        primary = -deviations
        chosen = _pick_tied_candidate(
            np.arange(size), primary, deviations, scores, sequences
        )
        return np.asarray([chosen], dtype=np.int64), np.empty(0, dtype=np.int64)

    length = len(str(sequences[0]))
    encoded = np.frombuffer(
        "".join(sequences.tolist()).encode("ascii"), dtype=np.uint8
    ).reshape(size, length)
    available = np.ones(size, dtype=bool)

    # Two farthest-point sweeps provide a deterministic approximate diameter
    # instead of anchoring the diverse set on the highest-scoring sequence.
    target_anchor = _pick_tied_candidate(
        np.arange(size), -deviations, deviations, scores, sequences
    )
    distances = np.count_nonzero(encoded != encoded[target_anchor], axis=1)
    first = _pick_tied_candidate(
        np.arange(size), distances, deviations, scores, sequences
    )

    selected = [first]
    insertion_distances: list[int] = []
    available[first] = False
    minimums = np.count_nonzero(encoded != encoded[first], axis=1)
    novelty = np.log1p(minimums.astype(np.float64))
    minimums[first] = -1
    while len(selected) < count:
        eligible = np.flatnonzero(available & (minimums >= minimum_distance))
        if not len(eligible):
            break
        chosen = _pick_tied_candidate(eligible, novelty, deviations, scores, sequences)
        insertion_distances.append(int(minimums[chosen]))
        selected.append(chosen)
        available[chosen] = False
        distances = np.count_nonzero(encoded != encoded[chosen], axis=1)
        minimums = np.minimum(minimums, distances)
        minimums[~available] = -1
        novelty += np.log1p(distances)
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(insertion_distances, dtype=np.int64),
    )


def _diversity_statistics(sequences: np.ndarray) -> dict[str, float | int]:
    if len(sequences) < 2:
        return {
            "minimum_pairwise_hamming": 0,
            "mean_pairwise_hamming": 0.0,
            "mean_nearest_neighbor_hamming": 0.0,
            "mean_ten_nearest_hamming": 0.0,
            "minimum_pairwise_hamming_fraction": 0.0,
            "mean_pairwise_hamming_fraction": 0.0,
            "mean_nearest_neighbor_hamming_fraction": 0.0,
            "mean_ten_nearest_hamming_fraction": 0.0,
        }
    length = len(str(sequences[0]))
    encoded = np.frombuffer(
        "".join(sequences.tolist()).encode("ascii"), dtype=np.uint8
    ).reshape(len(sequences), length)
    matrix = np.zeros((len(encoded), len(encoded)), dtype=np.int32)
    for index in range(len(encoded) - 1):
        distances = np.count_nonzero(encoded[index + 1 :] != encoded[index], axis=1)
        matrix[index, index + 1 :] = distances
        matrix[index + 1 :, index] = distances
    values = matrix[np.triu_indices(len(encoded), 1)].astype(np.float64)
    np.fill_diagonal(matrix, length + 1)
    neighbor_count = min(10, len(encoded) - 1)
    nearest = np.partition(matrix, neighbor_count - 1, axis=1)[:, :neighbor_count]
    minimum = int(values.min())
    mean = float(values.mean())
    mean_nearest = float(nearest.min(axis=1).mean())
    mean_ten_nearest = float(nearest.mean())
    return {
        "minimum_pairwise_hamming": minimum,
        "mean_pairwise_hamming": mean,
        "mean_nearest_neighbor_hamming": mean_nearest,
        "mean_ten_nearest_hamming": mean_ten_nearest,
        "minimum_pairwise_hamming_fraction": minimum / length,
        "mean_pairwise_hamming_fraction": mean / length,
        "mean_nearest_neighbor_hamming_fraction": mean_nearest / length,
        "mean_ten_nearest_hamming_fraction": mean_ten_nearest / length,
    }


def _blend_to_mean_floor(
    keys: np.ndarray,
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    fallback_sequences: np.ndarray,
    *,
    mean_floor: float,
    minimum_distance: int,
) -> tuple[np.ndarray, int]:
    """Replace as few diverse candidates as needed to meet a mean floor.

    The score-greedy fallback is already diversity-feasible.  Starting from
    the novelty-selected set, progressively remove its lowest-scoring members
    and refill from that fallback.  The first feasible blend therefore keeps
    the maximum possible number of novelty-selected sequences under this
    deterministic replacement order.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if (
        len(candidate_indices) == 0
        or scores[candidate_indices].mean() >= mean_floor - 1e-12
    ):
        return candidate_indices, 0

    positions = {str(key): index for index, key in enumerate(keys)}
    fallback_indices = np.asarray(
        [positions[str(sequence)] for sequence in fallback_sequences],
        dtype=np.int64,
    )
    union = np.unique(np.concatenate((candidate_indices, fallback_indices)))
    union_positions = {int(index): offset for offset, index in enumerate(union)}
    length = len(str(keys[union[0]]))
    encoded = np.frombuffer(
        "".join(keys[union].tolist()).encode("ascii"), dtype=np.uint8
    ).reshape(len(union), length)
    distances = np.zeros((len(union), len(union)), dtype=np.int32)
    for index in range(len(union) - 1):
        values = np.count_nonzero(encoded[index + 1 :] != encoded[index], axis=1)
        distances[index, index + 1 :] = values
        distances[index + 1 :, index] = values

    removal_order = sorted(
        candidate_indices.tolist(),
        key=lambda index: (scores[index], str(keys[index])),
    )
    fallback_order = sorted(
        fallback_indices.tolist(),
        key=lambda index: (-scores[index], str(keys[index])),
    )
    count = len(candidate_indices)
    for replacements in range(1, count + 1):
        removed = set(removal_order[:replacements])
        selected = [
            int(index) for index in candidate_indices if int(index) not in removed
        ]
        selected_set = set(selected)
        for index in fallback_order:
            if index in selected_set:
                continue
            local = union_positions[index]
            if minimum_distance and selected:
                selected_local = [union_positions[value] for value in selected]
                if np.any(distances[local, selected_local] < minimum_distance):
                    continue
            selected.append(index)
            selected_set.add(index)
            if len(selected) == count:
                break
        if (
            len(selected) == count
            and scores[selected].mean() >= mean_floor - 1e-12
        ):
            return np.asarray(selected, dtype=np.int64), replacements
    return candidate_indices, count


def _quality_diverse_snapshot(
    monitor: _SaturationMonitor,
    *,
    count: int,
    baseline: float,
    target: float,
    knee_gain: float,
    tolerance: float,
    minimum_distance: int,
    pool_size: int,
    seed: int,
    fallback: tuple[int, np.ndarray, np.ndarray, np.ndarray],
    mean_score_floor: float = -np.inf,
) -> tuple[tuple[int, np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """Select a diverse set near knee-fraction quality, reporting fallbacks."""
    keys = np.asarray(list(monitor.archive), dtype=str)
    scores = np.asarray([monitor.archive[key][0] for key in keys], dtype=np.float64)
    uncertainties = np.asarray(
        [monitor.archive[key][1] for key in keys], dtype=np.float64
    )
    deviations = np.abs(scores - target)
    # The tolerance is expressed as a fraction of baseline-to-knee gain. Flat
    # curves still receive a narrow absolute band rather than an empty pool.
    half_width = tolerance * max(abs(knee_gain), monitor.material_improvement)
    eligible = np.flatnonzero(deviations <= half_width + 1e-12)
    expanded = False
    if len(eligible) < count:
        expanded = True
        nearest_count = min(len(keys), max(count, 10 * count))
        eligible = np.argpartition(deviations, nearest_count - 1)[:nearest_count]

    # Bound the quadratic-in-output-count distance work for very long searches.
    # Uniform sampling inside the score band does not privilege high scores.
    sampled = False
    if len(eligible) > pool_size:
        sampled = True
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xD1A3]))
        closest_count = min(count, pool_size)
        closest = eligible[
            np.argpartition(deviations[eligible], closest_count - 1)[:closest_count]
        ]
        forced_keys = set(fallback[1].tolist())
        forced = np.asarray(
            [index for index in eligible if keys[index] in forced_keys],
            dtype=np.int64,
        )
        fixed = np.unique(np.concatenate((closest, forced)))
        if len(fixed) > pool_size:
            order = np.argsort(deviations[fixed], kind="stable")
            fixed = fixed[order[:pool_size]]
        remaining = np.setdiff1d(eligible, fixed, assume_unique=False)
        draw_count = pool_size - len(fixed)
        drawn = (
            rng.choice(remaining, size=draw_count, replace=False)
            if draw_count < len(remaining)
            else remaining
        )
        eligible = np.concatenate((fixed, drawn))

    local_indices, insertion_distances = _novelty_indices(
        keys[eligible],
        scores[eligible],
        target,
        count,
        minimum_distance,
    )
    if len(local_indices) < count and len(eligible) < len(keys):
        expanded = True
        expansion_limit = min(len(keys), pool_size)
        nearest_count = min(expansion_limit, max(10 * count, 2 * len(eligible)))
        while len(local_indices) < count:
            eligible = np.argpartition(deviations, nearest_count - 1)[:nearest_count]
            local_indices, insertion_distances = _novelty_indices(
                keys[eligible], scores[eligible], target, count, minimum_distance
            )
            if len(local_indices) == count or nearest_count == expansion_limit:
                break
            nearest_count = min(expansion_limit, 2 * nearest_count)

        # A bounded pool can miss a rare feasible combination. Include the
        # already-proven score-greedy set only as a final feasibility guard.
        if len(local_indices) < count and expansion_limit < len(keys):
            positions = {key: index for index, key in enumerate(keys)}
            forced = np.asarray(
                [positions[str(sequence)] for sequence in fallback[1]],
                dtype=np.int64,
            )
            eligible = np.unique(np.concatenate((eligible, forced)))
            local_indices, insertion_distances = _novelty_indices(
                keys[eligible],
                scores[eligible],
                target,
                count,
                minimum_distance,
            )
    selected_indices = eligible[local_indices]
    blend_replacements = 0
    if len(selected_indices) == count and np.isfinite(mean_score_floor):
        selected_indices, blend_replacements = _blend_to_mean_floor(
            keys,
            scores,
            selected_indices,
            np.asarray(fallback[1], dtype=str),
            mean_floor=mean_score_floor,
            minimum_distance=minimum_distance,
        )
    previous_statistics = _diversity_statistics(np.asarray(fallback[1], dtype=str))
    candidate_statistics = None
    used_fallback = len(selected_indices) < count
    fallback_reason = "infeasible_quality_pool" if used_fallback else None
    if not used_fallback:
        candidate_statistics = _diversity_statistics(keys[selected_indices])
        if scores[selected_indices].mean() < mean_score_floor - 1e-12:
            used_fallback = True
            fallback_reason = "selection_mean_below_relative_quality_floor"
    if not used_fallback:
        diversity_order = (
            "mean_ten_nearest_hamming",
            "mean_nearest_neighbor_hamming",
            "minimum_pairwise_hamming",
            "mean_pairwise_hamming",
        )
        candidate_key = tuple(candidate_statistics[name] for name in diversity_order)
        previous_key = tuple(previous_statistics[name] for name in diversity_order)
        if candidate_key < previous_key:
            used_fallback = True
            fallback_reason = "no_local_diversity_improvement"
    if used_fallback:
        query_count, selected_sequences, selected_scores, selected_uncertainties = (
            fallback
        )
    else:
        selected_sequences = keys[selected_indices]
        selected_scores = scores[selected_indices]
        selected_uncertainties = uncertainties[selected_indices]
        output_order = np.lexsort((selected_sequences, -selected_scores))
        selected_sequences = selected_sequences[output_order]
        selected_scores = selected_scores[output_order]
        selected_uncertainties = selected_uncertainties[output_order]
        query_count = max(
            monitor.discovery_queries[str(sequence)] for sequence in selected_sequences
        )
    selected_scores = np.asarray(selected_scores, dtype=np.float64)
    gain_scale = max(abs(knee_gain), monitor.material_improvement)
    fractions = (selected_scores - baseline) / gain_scale
    metadata: dict[str, Any] = {
        "quality_diversity_pool_count": int(len(eligible)),
        "quality_diversity_pool_sampled": sampled,
        "quality_diversity_method": "greedy_log_distance_product",
        "quality_band_expanded": expanded,
        "quality_diversity_fallback": used_fallback,
        "quality_diversity_fallback_reason": fallback_reason,
        "quality_floor_blend_replacements": int(blend_replacements),
        "requested_selection_mean_floor": (
            float(mean_score_floor) if np.isfinite(mean_score_floor) else None
        ),
        "selection_mean_floor_satisfied": bool(
            selected_scores.mean() >= mean_score_floor - 1e-12
        ),
        "selection_score_half_width": float(half_width),
        "effective_selection_score_half_width": float(
            np.max(deviations[eligible]) if len(eligible) else 0.0
        ),
        "selected_score_mean": float(selected_scores.mean()),
        "selected_score_minimum": float(selected_scores.min()),
        "selected_score_maximum": float(selected_scores.max()),
        "selected_normalized_gain_minimum": float(fractions.min()),
        "selected_normalized_gain_mean": float(fractions.mean()),
        "selected_normalized_gain_maximum": float(fractions.max()),
        "score_greedy_score_mean": float(np.mean(fallback[2])),
        **{
            f"score_greedy_{name}": value for name, value in previous_statistics.items()
        },
        **(
            {
                f"quality_diversity_candidate_{name}": value
                for name, value in candidate_statistics.items()
            }
            if candidate_statistics is not None
            else {}
        ),
        **_diversity_statistics(np.asarray(selected_sequences, dtype=str)),
    }
    if len(insertion_distances) and not used_fallback:
        metadata["novelty_selection_last_minimum_distance"] = int(
            insertion_distances[-1]
        )
    return (
        (
            int(query_count),
            np.asarray(selected_sequences, dtype=str),
            selected_scores,
            np.asarray(selected_uncertainties, dtype=np.float64),
        ),
        metadata,
    )


def _optimize_with_saturation(
    scorer: SequenceScorer,
    length: int,
    *,
    algorithm: str,
    alphabet: str = "ACGT",
    num_candidates: int = 100,
    initial=None,
    objective: TargetObjective
    | MultiTaskObjective
    | list[str]
    | tuple[str, ...]
    | None = None,
    constraint: ConstraintSpec | None = None,
    saturation: SaturationConfig,
    score_threshold: float | None = None,
    seed: int = 0,
    count_repeated: bool = False,
    cds_reference: str | None = None,
    **optimizer_parameters: Any,
) -> SaturationResult:
    """Run until a plateau and return q-fraction candidate sets.

    A q-fraction result is the earliest archive whose top-candidate mean reaches
    that fraction of the gain from the first valid archive to the observed plateau.
    """
    from .algorithms import (
        _REGISTRY,
        _apply_fixed_region,
        _constraint_initial,
        _configured_space,
        _initial,
        _prediction_initial,
        available_algorithms,
    )

    key = str(algorithm).strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown algorithm {algorithm!r}; choose from {', '.join(available_algorithms())}"
        )
    options = saturation
    minimum, patience, maximum = options.resolve(
        num_candidates, None if initial is None else len(initial)
    )
    default_budget = max(1000 * num_candidates, minimum + patience)
    if key == "simulated_annealing" and maximum is None:
        default_budget = max(
            default_budget,
            (int(optimizer_parameters.get("annealing_iterations", 1000)) + 1)
            * max(num_candidates, 1),
        )
    config = OptimizationConfig(
        budget=maximum or default_budget,
        num_candidates=num_candidates,
        **optimizer_parameters,
    )
    rng = np.random.default_rng(seed)
    constraint_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0xC0DEC0DE])
    )
    space = _configured_space(
        length,
        alphabet,
        config,
        key,
        initial,
        cds_reference=cds_reference,
    )
    random.seed(seed)
    objective_scorer = ObjectiveScorer(
        scorer,
        TargetObjective() if objective is None else objective,
    )
    space = _apply_fixed_region(space, config.fixed_region, initial, rng)
    prepared = _initial(initial, space, config, rng)
    if options.plateau_patience_queries is None:
        # Large proposal batches must not turn a population-scaled window into
        # only one or two observations. Explicit user windows remain untouched.
        batch_size = len(prepared)
        if key == "basesearch":
            batch_size = config.basesearch_batch_size
        elif key == "cross_entropy_method":
            batch_size = max(config.cross_entropy_batch_size, num_candidates)
        elif key in {"beam_search", "ism_greedy"}:
            mutable = len(getattr(space, "mutable_positions", range(space.length)))
            neighbors = int(
                getattr(
                    space, "single_mutant_count", mutable * (space.alphabet_size - 1)
                )
            )
            parents = (
                max(1, int(round(num_candidates * config.elite_fraction)))
                if key == "beam_search"
                else len(prepared)
            )
            batch_size = parents * neighbors
        patience = max(patience, 3 * batch_size)
        if key == "simulated_annealing":
            patience *= 5
    monitor = _SaturationMonitor(
        num_candidates,
        minimum,
        patience,
        options.material_improvement,
        config.diversity,
        options.relative_material_improvement,
        score_threshold=score_threshold,
        patience_rounds=(options.plateau_patience_rounds if options.plateau_patience_queries is None else None),
        report_progress=getattr(scorer, "report_progress", None),
    )
    proposal_count_algorithms = {
        "ledidi",
        "vanilla_activation_maximization",
        "fast_seqprop",
        "den",
    }
    budgeted = _TrackingScorer(
        objective_scorer,
        maximum,
        count_repeated=count_repeated or key in proposal_count_algorithms,
        constraint=constraint,
        constraint_rng=constraint_rng,
        monitor=monitor,
    )
    stop_reason = "algorithm_stopped"
    try:
        prepared = _constraint_initial(prepared, budgeted, space, config, rng)
        prepared = _prediction_initial(prepared, budgeted, space, config, rng)
        algorithm_result = _REGISTRY[key](prepared, budgeted, space, config, rng)
        if algorithm_result is not None and algorithm_result.metadata.get("stop_reason") == "lineage_saturation":
            stop_reason = "plateau"
        if objective_scorer.target_stop_reached:
            stop_reason = "target_reached"
        elif maximum is not None and budgeted.query_count >= maximum:
            stop_reason = "maximum_queries"
    except _ScoreThresholdReached:
        stop_reason = "target_reached"
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
    baseline, plateau = float(monitor.initial_mean), float(top_means[-1])
    if score_threshold is not None:
        # The threshold applies to the best feasible score. A complete
        # candidate archive is still required for the requested output count.
        index = max(
            range(len(monitor.snapshots)),
            key=lambda i: (float(monitor.snapshots[i][2].max()), i),
        )
        result = _selection_result(
            key, monitor.snapshots[index], fraction=None,
            total_queries=budgeted.query_count, knee_queries=0,
            baseline=baseline, plateau=plateau, objective_scorer=objective_scorer,
            budgeted=budgeted, trajectory=[], stop_reason=stop_reason,
            diversity=config.diversity,
        )
        result.metadata.update(
            selection="score_threshold", score_threshold=float(score_threshold),
            threshold_reached=stop_reason == "target_reached",
            candidates_at_threshold=int(np.count_nonzero(result.scores >= score_threshold)),
        )
        return SaturationResult({}, budgeted.query_count, 0, stop_reason, [],
                                selection="threshold", threshold_result=result)
    # The initial population is a curve anchor, never an undersized output.
    anchored_queries = np.concatenate(([monitor.initial_query], query_counts))
    anchored_means = np.concatenate(([baseline], top_means))
    knee_index = max(
        0,
        _knee_index(anchored_queries, anchored_means, options.material_improvement) - 1,
    )
    if np.max(top_means) <= baseline + options.material_improvement:
        knee_index = int(np.argmax(top_means))
    peak_index = int(np.argmax(top_means))
    observed_gain = max(0.0, float(top_means[peak_index]) - baseline)
    if options.relaxed_knee and observed_gain > options.material_improvement:
        eligible_knees = np.flatnonzero(
            top_means
            >= baseline + options.knee_minimum_gain_fraction * observed_gain
        )
        x = (query_counts - monitor.initial_query) / max(
            float(query_counts[-1] - monitor.initial_query), 1.0
        )
        distances = (top_means - baseline) / observed_gain - x
        best = int(eligible_knees[np.argmax(distances[eligible_knees])])
        knee_index = best if distances[best] > 1e-12 else int(eligible_knees[0])
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
        index = _first_eligible_index(top_means, target)
        selections[float(fraction)] = _selection_result(
            key,
            monitor.snapshots[index],
            fraction=float(fraction),
            total_queries=budgeted.query_count,
            knee_queries=knee_queries,
            baseline=baseline,
            plateau=plateau,
            objective_scorer=objective_scorer,
            budgeted=budgeted,
            trajectory=trajectory,
            stop_reason=stop_reason,
            diversity=config.diversity,
        )
        selections[float(fraction)].metadata.update(
            selected_top_mean=float(top_means[index]),
            target_top_mean=float(target),
            minimum_queries=minimum,
            plateau_patience_queries=patience,
            maximum_queries=maximum,
            initial_population_size=len(prepared),
            material_improvement_floor=options.material_improvement,
            relative_material_improvement=options.relative_material_improvement,
            final_material_improvement_threshold=monitor.last_material_threshold,
        )
    knee = _selection_result(
        key,
        monitor.snapshots[knee_index],
        fraction=None,
        total_queries=budgeted.query_count,
        knee_queries=knee_queries,
        baseline=baseline,
        plateau=plateau,
        objective_scorer=objective_scorer,
        budgeted=budgeted,
        trajectory=trajectory,
        stop_reason=stop_reason,
        diversity=config.diversity,
    )
    knee.metadata.update(
        minimum_queries=minimum,
        plateau_patience_queries=patience,
        plateau_patience_rounds=monitor.patience_rounds,
        observed_rounds=monitor.observation_round,
        maximum_queries=maximum,
        initial_population_size=len(prepared),
        material_improvement_floor=options.material_improvement,
        relative_material_improvement=options.relative_material_improvement,
        final_material_improvement_threshold=monitor.last_material_threshold,
        plateau_confirmed=stop_reason == "plateau",
        knee_method=(
            "late_gain_constrained_maximum_normalized_distance"
            if options.relaxed_knee
            else "maximum_normalized_distance_above_chord"
        ),
    )
    knee_mean = float(top_means[knee_index])
    knee_gain = knee_mean - baseline
    knee_target = baseline + options.knee_fraction * knee_gain
    if options.relaxed_knee:
        # The relaxed elbow already supplies the quality target. Applying the
        # historical 0.9 discount again would compound two relaxations and can
        # produce unnecessarily low-scoring outputs.
        knee_target = knee_mean
    selected_index = _first_eligible_index(top_means[: knee_index + 1], knee_target)
    selection_floor = (
        baseline + options.selection_minimum_gain_fraction * observed_gain
        if options.relaxed_knee and observed_gain > options.material_improvement
        else -np.inf
    )
    diverse_snapshot, diversity_metadata = _quality_diverse_snapshot(
        monitor,
        count=num_candidates,
        baseline=baseline,
        target=knee_target,
        knee_gain=observed_gain if options.relaxed_knee else knee_gain,
        tolerance=options.knee_fraction_tolerance,
        minimum_distance=config.diversity,
        pool_size=options.diversity_pool_size,
        seed=seed,
        # The peak-mean archive is the strongest known diversity-feasible fallback.
        # It is used only when the requested relative quality floor and the
        # minimum Hamming distance cannot be satisfied simultaneously.
        fallback=(
            monitor.snapshots[peak_index]
            if options.relaxed_knee
            else monitor.snapshots[selected_index]
        ),
        mean_score_floor=selection_floor,
    )
    knee_fraction_result = _selection_result(
        key,
        diverse_snapshot,
        fraction=options.knee_fraction,
        total_queries=budgeted.query_count,
        knee_queries=knee_queries,
        baseline=baseline,
        plateau=plateau,
        objective_scorer=objective_scorer,
        budgeted=budgeted,
        trajectory=trajectory,
        stop_reason=stop_reason,
        diversity=config.diversity,
    )
    knee_fraction_result.metadata.update(
        selection="quality_diversity_near_knee_gain",
        knee_top_mean=knee_mean,
        relaxed_knee=options.relaxed_knee,
        observed_peak_gain=observed_gain,
        knee_minimum_gain_fraction=(
            options.knee_minimum_gain_fraction if options.relaxed_knee else None
        ),
        selection_gain_floor=(
            options.selection_minimum_gain_fraction if options.relaxed_knee else None
        ),
        selection_target_mean=knee_target,
        minimum_queries=minimum,
        plateau_patience_queries=patience,
        plateau_patience_rounds=monitor.patience_rounds,
        observed_rounds=monitor.observation_round,
        maximum_queries=maximum,
        initial_population_size=len(prepared),
        plateau_confirmed=stop_reason == "plateau",
        knee_fraction_tolerance=options.knee_fraction_tolerance,
        diversity_pool_size=options.diversity_pool_size,
        parent_pool_multiplier=config.parent_pool_multiplier,
        **({"offspring_comparison": "rollout_root",
            "rollout": "accepted_child", "rollout_max_depth": config.adalead_max_rollout,
            "rollout_expected_mutations": config.adalead_mu, "rollout_allow_equal": config.adalead_allow_equal,
            "rollout_lineages": config.adalead_lineages or len(prepared),
            "parent_selection": "rank_weighted_quality_with_novelty_exploration",
            "survivor_selection": "rank_weighted_quality_with_novelty_exploration",
            "exploration_fraction": config.adalead_exploration_fraction,
            "quality_rank_power": config.adalead_quality_power,
            "best_anchor_preserved": True}
           if key == "adalead" else {}),
        **({"diversity_penalty": "nearest_neighbor_similarity"} if key == "den" else {}),
        **({
            "iterations": max(0, monitor.observation_round - 1),
            "maximum_iterations": config.annealing_iterations,
            "initial_temperature": config.annealing_initial_temperature,
            "final_temperature": config.annealing_final_temperature,
            "temperature_schedule": "linear_by_iteration",
            "last_temperature": (
                config.annealing_initial_temperature
                if monitor.observation_round <= 1
                else config.annealing_initial_temperature
                + min(
                    (monitor.observation_round - 2)
                    / max(config.annealing_iterations - 1, 1),
                    1.0,
                )
                * (
                    config.annealing_final_temperature
                    - config.annealing_initial_temperature
                )
            ),
        } if key == "simulated_annealing" else {}),
        **({"parent_selection": ("quality_progress_weighted_clusters" if key == "basesearch" else "cluster_round_robin"), "parent_cluster_cap": 10,
            "parent_cluster_diameter_fraction": 0.2}
           if key in {"beam_search", "basesearch"} else {}),
        material_improvement_floor=options.material_improvement,
        relative_material_improvement=options.relative_material_improvement,
        final_material_improvement_threshold=monitor.last_material_threshold,
        baseline_query_count=monitor.initial_query,
        baseline_population_mean=baseline,
        first_complete_query_count=int(query_counts[0]),
        selection_target_reached=bool(diverse_snapshot[2].mean() >= knee_target - 1e-12),
        **diversity_metadata,
    )
    if key == "adalead":
        fields = ("parent_selection", "survivor_selection", "exploration_fraction",
                  "quality_rank_power", "best_anchor_preserved", "offspring_comparison",
                  "rollout", "rollout_max_depth", "rollout_expected_mutations",
                  "rollout_allow_equal", "rollout_lineages", "parent_pool_multiplier")
        for result in [*selections.values(), knee]:
            result.metadata.update({name: knee_fraction_result.metadata[name] for name in fields})
    if key == "ism_greedy" and hasattr(budgeted, "ism_diagnostics"):
        for result in [*selections.values(), knee, knee_fraction_result]:
            result.metadata["ism_lineage_diagnostics"] = budgeted.ism_diagnostics
    fixed_positions = tuple(getattr(space, "fixed_positions", ()))
    if fixed_positions:
        for result in [*selections.values(), knee, knee_fraction_result]:
            result.metadata["fixed_positions"] = list(fixed_positions)
    return SaturationResult(
        selections=selections,
        total_query_count=budgeted.query_count,
        knee_query_count=knee_queries,
        stop_reason=stop_reason,
        trajectory=trajectory,
        knee=knee,
        selection=options.selection,
        knee_fraction_result=knee_fraction_result,
    )
