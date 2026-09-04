"""Dataset-independent discrete sequence optimization algorithms."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from .api import CachedScorer, OptimizationConfig, OptimizationResult, ScoreBatch, SequenceScorer, SequenceSpace
from .objectives import ObjectiveScorer, TargetObjective

Optimizer = Callable[[np.ndarray, CachedScorer, SequenceSpace, OptimizationConfig, np.random.Generator], OptimizationResult]
_REGISTRY: dict[str, Optimizer] = {}


def register_algorithm(name: str):
    """Register an optimizer function under a stable user-facing name."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("algorithm name must not be empty")

    def decorator(function: Optimizer) -> Optimizer:
        if key in _REGISTRY:
            raise ValueError(f"algorithm already registered: {key}")
        _REGISTRY[key] = function
        return function

    return decorator


def available_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _record(archive: dict[str, tuple[float, float]], sequences: np.ndarray, batch: ScoreBatch) -> None:
    uncertainty = np.zeros(len(batch.scores)) if batch.uncertainties is None else batch.uncertainties
    for sequence, score, std in zip(sequences, batch.scores, uncertainty):
        key = str(sequence)
        if key not in archive or float(score) > archive[key][0]:
            archive[key] = (float(score), float(std))


def _score(scorer: CachedScorer, archive: dict[str, tuple[float, float]], sequences: np.ndarray) -> ScoreBatch:
    batch = scorer.score(sequences)
    _record(archive, sequences, batch)
    return batch


def _finalize(name: str, archive: dict[str, tuple[float, float]], config: OptimizationConfig, scorer: CachedScorer, metadata: dict[str, Any], trajectory: list[dict[str, float | int]]) -> OptimizationResult:
    if len(archive) < config.num_candidates:
        raise RuntimeError(f"only {len(archive)} unique sequences were scored; {config.num_candidates} required")
    ranked = sorted(archive.items(), key=lambda item: (-item[1][0], item[0]))[: config.num_candidates]
    return OptimizationResult(
        name,
        np.asarray([item[0] for item in ranked], dtype=str),
        np.asarray([item[1][0] for item in ranked], dtype=np.float64),
        np.asarray([item[1][1] for item in ranked], dtype=np.float64),
        scorer.query_count,
        metadata,
        trajectory,
    )


def _initial(initial, space: SequenceSpace, config: OptimizationConfig, rng: np.random.Generator) -> np.ndarray:
    if initial is None:
        capacity = space.alphabet_size ** space.length
        if config.num_candidates > capacity:
            raise ValueError(
                f"cannot draw {config.num_candidates} unique seeds from a space of size {capacity}"
            )
        seeds: dict[str, None] = {}
        while len(seeds) < config.num_candidates:
            needed = config.num_candidates - len(seeds)
            for sequence in space.random(min(max(64, needed), 4096), rng):
                seeds.setdefault(str(sequence), None)
                if len(seeds) == config.num_candidates:
                    break
        values = np.asarray(list(seeds), dtype=str)
    else:
        values = space.validate(initial)
    if len(values) != config.num_candidates:
        raise ValueError(f"initial must contain exactly {config.num_candidates} sequences")
    if len(np.unique(values)) != len(values):
        raise ValueError("initial sequences must be unique")
    return values.copy()


@register_algorithm("random")
def random_search(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    _score(scorer, archive, initial)
    while scorer.remaining is None or scorer.remaining > 0:
        count = min(4096, scorer.remaining or 4096)
        before = scorer.query_count
        _score(scorer, archive, space.random(count, rng))
        if scorer.query_count == before:
            break
    return _finalize("random", archive, config, scorer, {"strategy": "uniform random screening"}, [])


@register_algorithm("ism_greedy")
def ism_greedy(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    current = initial.copy(); current_scores = _score(scorer, archive, current).scores
    trajectory = [{"query_count": scorer.query_count, "best_score": float(current_scores.max()), "mean_score": float(current_scores.mean())}]
    block = space.length * (space.alphabet_size - 1)
    while scorer.remaining is None or scorer.remaining >= block:
        before = scorer.query_count
        order = rng.permutation(len(current))
        count = len(order) if scorer.remaining is None else min(len(order), scorer.remaining // block)
        chosen = order[:count]
        mutants = space.single_mutants(current[chosen])
        batch = _score(scorer, archive, mutants)
        for offset, index in enumerate(chosen):
            values = batch.scores[offset * block : (offset + 1) * block]
            best = int(np.argmax(values))
            if values[best] > current_scores[index]:
                current[index] = mutants[offset * block + best]
                current_scores[index] = values[best]
        trajectory.append({"query_count": scorer.query_count, "best_score": float(current_scores.max()), "mean_score": float(current_scores.mean())})
        if not len(chosen) or scorer.query_count == before:
            break
    return _finalize("ism_greedy", archive, config, scorer, {"sweeps": len(trajectory) - 1}, trajectory)


@register_algorithm("simulated_annealing")
def simulated_annealing(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    current = initial.copy(); current_scores = _score(scorer, archive, current).scores
    temperature = max(float(np.std(current_scores)), 1e-3)
    trajectory = []
    while scorer.remaining is None or scorer.remaining >= len(current):
        before = scorer.query_count
        proposals = space.mutate_one(current, rng); values = _score(scorer, archive, proposals).scores
        progress = scorer.query_count / max(config.budget, 1)
        temp = max(temperature * (0.001 ** progress), 1e-6)
        accept = (values >= current_scores) | (rng.random(len(current)) < np.exp(np.minimum((values - current_scores) / temp, 0)))
        current[accept] = proposals[accept]; current_scores[accept] = values[accept]
        trajectory.append({"query_count": scorer.query_count, "best_score": float(current_scores.max()), "mean_score": float(current_scores.mean())})
        if scorer.query_count == before:
            break
    return _finalize("simulated_annealing", archive, config, scorer, {"initial_temperature": temperature}, trajectory)


@register_algorithm("genetic_algorithm")
def genetic_algorithm(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    population = initial.copy(); scores = _score(scorer, archive, population).scores
    elite_count = max(1, int(round(len(population) * config.elite_fraction))); trajectory = []
    while scorer.remaining is None or scorer.remaining >= len(population):
        before = scorer.query_count
        elite = population[np.argsort(scores)[-elite_count:]]
        a = space.encode(elite[rng.integers(len(elite), size=len(population))]); b = space.encode(elite[rng.integers(len(elite), size=len(population))])
        child = np.where(rng.random(a.shape) < 0.5, a, b); mask = rng.random(child.shape) < config.mutation_rate; child = np.where(mask, rng.integers(space.alphabet_size, size=child.shape), child)
        children = space.decode(child); child_scores = _score(scorer, archive, children).scores
        combined = np.concatenate([population, children]); combined_scores = np.concatenate([scores, child_scores]); keep = np.argsort(combined_scores)[-len(population):]; population, scores = combined[keep], combined_scores[keep]
        trajectory.append({"query_count": scorer.query_count, "best_score": float(scores.max()), "mean_score": float(scores.mean())})
        if scorer.query_count == before:
            break
    return _finalize("genetic_algorithm", archive, config, scorer, {"generations": len(trajectory), "elite_count": elite_count}, trajectory)


@register_algorithm("adalead")
def adalead(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    population = initial.copy(); scores = _score(scorer, archive, population).scores; trajectory = []
    while scorer.remaining is None or scorer.remaining >= len(population):
        before = scorer.query_count
        elite_count = max(1, int(round(len(population) * config.elite_fraction))); elite = population[np.argsort(scores)[-elite_count:]]; parents = elite[rng.integers(len(elite), size=len(population))]; proposals = space.mutate_one(parents, rng); values = _score(scorer, archive, proposals).scores; improved = values > scores; population[improved] = proposals[improved]; scores[improved] = values[improved]
        trajectory.append({"query_count": scorer.query_count, "best_score": float(scores.max()), "mean_score": float(scores.mean())})
        if scorer.query_count == before:
            break
    return _finalize("adalead", archive, config, scorer, {"steps": len(trajectory)}, trajectory)


def run(
    name: str,
    scorer: SequenceScorer,
    space: SequenceSpace,
    config: OptimizationConfig,
    *,
    initial: Sequence[str] | np.ndarray | None = None,
    objective: TargetObjective | None = None,
    seed: int = 0,
    count_repeated: bool = False,
) -> OptimizationResult:
    """Run one algorithm against a single- or multi-task activity scorer.

    The model output is scalarized by ``objective``. If omitted, output zero is
    maximized, which is the natural default for a single-task model.
    """
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown algorithm {name!r}; choose from {', '.join(available_algorithms())}")
    rng = np.random.default_rng(seed); random.seed(seed)
    prepared = _initial(initial, space, config, rng)
    objective_scorer: ObjectiveScorer | None
    if isinstance(scorer, CachedScorer):
        if objective is not None:
            raise ValueError("pass an uncached activity scorer when using a multi-task objective")
        budgeted = scorer
        objective_scorer = None
    else:
        objective_scorer = ObjectiveScorer(scorer, objective or TargetObjective())
        budgeted = CachedScorer(objective_scorer, config.budget, count_repeated=count_repeated)
    if budgeted.budget is None:
        budgeted.budget = config.budget
    result = _REGISTRY[key](prepared, budgeted, space, config, rng)
    if objective_scorer is not None:
        activities = objective_scorer.activities_for(result.sequences)
        result.attach_activities(
            activities,
            objective_scorer.resolved_task_names(activities.shape[1]),
        )
        result.metadata["objective"] = objective_scorer.objective.describe(
            activities.shape[1], objective_scorer.task_names
        )
    return result


def run_many(
    names: Iterable[str],
    scorer: SequenceScorer,
    space: SequenceSpace,
    config: OptimizationConfig,
    *,
    initial: Sequence[str] | np.ndarray | None = None,
    objective: TargetObjective | None = None,
    seed: int = 0,
) -> dict[str, OptimizationResult]:
    """Run algorithms independently; each receives a fresh budget/cache."""
    return {
        str(name): run(
            name,
            scorer,
            space,
            config,
            initial=initial,
            objective=objective,
            seed=seed + offset,
        )
        for offset, name in enumerate(names)
    }


def optimize(
    scorer: SequenceScorer,
    space: SequenceSpace,
    *,
    algorithm: str,
    budget: int,
    num_candidates: int = 100,
    initial: Sequence[str] | np.ndarray | None = None,
    objective: TargetObjective | None = None,
    seed: int = 0,
    mutation_rate: float = 0.02,
    elite_fraction: float = 0.2,
    learning_rate: float = 0.1,
    count_repeated: bool = False,
) -> OptimizationResult:
    """Convenience interface that constructs :class:`OptimizationConfig`."""
    config = OptimizationConfig(
        budget=budget,
        num_candidates=num_candidates,
        mutation_rate=mutation_rate,
        elite_fraction=elite_fraction,
        learning_rate=learning_rate,
    )
    return run(
        algorithm,
        scorer,
        space,
        config,
        initial=initial,
        objective=objective,
        seed=seed,
        count_repeated=count_repeated,
    )


def random_optimize(scorer: SequenceScorer, space: SequenceSpace, **kwargs) -> OptimizationResult:
    return optimize(scorer, space, algorithm="random", **kwargs)


def ism_optimize(scorer: SequenceScorer, space: SequenceSpace, **kwargs) -> OptimizationResult:
    return optimize(scorer, space, algorithm="ism_greedy", **kwargs)


def simulated_annealing_optimize(scorer: SequenceScorer, space: SequenceSpace, **kwargs) -> OptimizationResult:
    return optimize(scorer, space, algorithm="simulated_annealing", **kwargs)


def genetic_algorithm_optimize(scorer: SequenceScorer, space: SequenceSpace, **kwargs) -> OptimizationResult:
    return optimize(scorer, space, algorithm="genetic_algorithm", **kwargs)


def adalead_optimize(scorer: SequenceScorer, space: SequenceSpace, **kwargs) -> OptimizationResult:
    return optimize(scorer, space, algorithm="adalead", **kwargs)


# Requested spelling retained as a public alias; lower-case ``ism_optimize`` is
# preferred in Python code.
ISM_optimize = ism_optimize
