"""Dataset-independent discrete sequence optimization algorithms."""

from __future__ import annotations

import inspect
import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from .api import (
    _sequence_space,
    CachedScorer,
    FixedRegionSequenceSpace,
    OptimizationConfig,
    OptimizationResult,
    ScoreBatch,
    SequenceScorer,
    SequenceSpace,
    normalize_nucleotide_sequences,
    resolve_fixed_positions,
)
from .constraints import ConstraintSpec
from .objectives import MultiTaskObjective, ObjectiveScorer, TargetObjective
from .saturation import SaturationConfig, SaturationResult

Optimizer = Callable[
    [np.ndarray, CachedScorer, SequenceSpace, OptimizationConfig, np.random.Generator],
    OptimizationResult,
]
_REGISTRY: dict[str, Optimizer] = {}
_DEFAULT_BUDGET = 10_000


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


def _record(
    archive: dict[str, tuple[float, float]], sequences: np.ndarray, batch: ScoreBatch
) -> None:
    uncertainty = (
        np.zeros(len(batch.scores))
        if batch.uncertainties is None
        else batch.uncertainties
    )
    for sequence, score, std in zip(sequences, batch.scores, uncertainty):
        key = str(sequence)
        if key not in archive or float(score) > archive[key][0]:
            archive[key] = (float(score), float(std))


def _score(
    scorer: CachedScorer, archive: dict[str, tuple[float, float]], sequences: np.ndarray
) -> ScoreBatch:
    batch = scorer.score(sequences)
    passed = scorer.hard_constraints_passed(sequences)
    if passed.any():
        _record(
            archive,
            np.asarray(sequences)[passed],
            ScoreBatch(
                batch.scores[passed],
                None if batch.uncertainties is None else batch.uncertainties[passed],
            ),
        )
    return batch


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def _rank_archive(
    archive: dict[str, tuple[float, float]],
    scorer: CachedScorer,
) -> list[tuple[str, tuple[float, float]]]:
    sequences = np.asarray(list(archive), dtype=str)
    scores = scorer.current_scores(sequences)
    uncertainties = scorer.current_uncertainties(sequences)
    ranked = [
        (sequence, (float(score), float(uncertainty)))
        for sequence, score, uncertainty in zip(sequences, scores, uncertainties)
    ]
    return sorted(ranked, key=lambda item: (-item[1][0], item[0]))


def _select_diverse(
    ranked: Sequence[tuple[str, tuple[float, float]]],
    count: int,
    diversity: int,
) -> list[tuple[str, tuple[float, float]]]:
    if count == 1 or diversity == 0:
        return list(ranked[:count])
    selected: list[tuple[str, tuple[float, float]]] = []
    for item in ranked:
        sequence = item[0]
        if all(
            _hamming_distance(sequence, chosen[0]) >= diversity for chosen in selected
        ):
            selected.append(item)
            if len(selected) == count:
                break
    return selected


def _finalize(
    name: str,
    archive: dict[str, tuple[float, float]],
    config: OptimizationConfig,
    scorer: CachedScorer,
    metadata: dict[str, Any],
    trajectory: list[dict[str, float | int]],
) -> OptimizationResult:
    if len(archive) < config.num_candidates:
        raise RuntimeError(
            f"only {len(archive)} unique sequences were scored; {config.num_candidates} required"
        )
    ranked = _select_diverse(
        _rank_archive(archive, scorer),
        config.num_candidates,
        config.diversity,
    )
    if len(ranked) < config.num_candidates:
        raise RuntimeError(
            f"only {len(ranked)} sequences satisfy diversity={config.diversity}; "
            "increase budget or lower diversity"
        )
    metadata = {
        **metadata,
        "diversity": config.diversity,
        "stop_reason": (
            "target_reached"
            if bool(getattr(scorer.scorer, "target_stop_reached", False))
            else metadata.get("stop_reason", "budget")
        ),
    }
    return OptimizationResult(
        name,
        np.asarray([item[0] for item in ranked], dtype=str),
        np.asarray([item[1][0] for item in ranked], dtype=np.float64),
        np.asarray([item[1][1] for item in ranked], dtype=np.float64),
        scorer.query_count,
        metadata,
        trajectory,
    )


def _initial(
    initial,
    space: SequenceSpace,
    config: OptimizationConfig,
    rng: np.random.Generator,
    *,
    anchor: str | None = None,
) -> np.ndarray:
    maximum_distance = int(getattr(space, "max_hamming_distance", space.length))
    if config.num_candidates > 1 and config.diversity > maximum_distance:
        raise ValueError(
            f"diversity={config.diversity} exceeds the {maximum_distance} mutable "
            "positions available for pairwise differences; "
            "lower diversity or request one candidate"
        )
    if initial is None:
        capacity = getattr(space, "capacity", space.alphabet_size**space.length)
        if config.num_candidates > capacity:
            raise ValueError(
                f"cannot draw {config.num_candidates} unique seeds from a space of size {capacity}"
            )
        seeds: dict[str, None] = {}
        if anchor is not None:
            anchor_value = str(space.validate([anchor])[0])
            seeds[anchor_value] = None
        draws = 0
        maximum_draws = max(10_000, 1000 * config.num_candidates)
        while len(seeds) < config.num_candidates and draws < maximum_draws:
            needed = config.num_candidates - len(seeds)
            candidates = space.random(min(max(64, needed), 4096), rng)
            draws += len(candidates)
            for sequence in candidates:
                value = str(sequence)
                if config.num_candidates == 1 or all(
                    _hamming_distance(value, chosen) >= config.diversity
                    for chosen in seeds
                ):
                    seeds.setdefault(value, None)
                if len(seeds) == config.num_candidates:
                    break
        if len(seeds) < config.num_candidates:
            raise ValueError(
                f"cannot draw {config.num_candidates} seeds satisfying "
                f"diversity={config.diversity}; lower diversity or supply initial"
            )
        values = np.asarray(list(seeds), dtype=str)
    else:
        values = space.validate(initial)
    if len(values) != config.num_candidates:
        raise ValueError(
            f"initial must contain exactly {config.num_candidates} sequences"
        )
    if len(np.unique(values)) != len(values):
        raise ValueError("initial sequences must be unique")
    return values.copy()


def _constraint_initial(
    initial: np.ndarray,
    scorer: CachedScorer,
    space: SequenceSpace,
    config: OptimizationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if scorer.constraint_evaluator is None:
        return initial
    passed = scorer.constraints_passed(initial)
    accepted = [str(sequence) for sequence, keep in zip(initial, passed) if keep]
    seen = {str(sequence) for sequence in initial}
    capacity = getattr(space, "capacity", space.alphabet_size**space.length)
    maximum_unique = min(capacity, max(10_000, 1000 * config.num_candidates))
    draws = 0
    maximum_draws = max(10_000, 10 * maximum_unique)
    while (
        len(accepted) < config.num_candidates
        and len(seen) < maximum_unique
        and draws < maximum_draws
    ):
        count = min(max(64, config.num_candidates - len(accepted)), 4096)
        candidates = space.random(count, rng)
        draws += len(candidates)
        novel = np.asarray(
            list(
                dict.fromkeys(
                    str(sequence)
                    for sequence in candidates
                    if str(sequence) not in seen
                )
            ),
            dtype=str,
        )
        seen.update(map(str, novel))
        if not len(novel):
            continue
        keep = scorer.constraints_passed(novel)
        accepted.extend(str(sequence) for sequence, value in zip(novel, keep) if value)
    if len(accepted) < config.num_candidates:
        raise ValueError(
            "constraints did not admit enough unique initial sequences; "
            "supply less restrictive constraints or feasible initial sequences"
        )
    return np.asarray(accepted[: config.num_candidates], dtype=str)


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
    return _finalize(
        "random", archive, config, scorer, {"strategy": "uniform random screening"}, []
    )


@register_algorithm("ism_greedy")
def ism_greedy(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    current = initial.copy()
    current_scores = _score(scorer, archive, current).scores
    trajectory = [
        {
            "query_count": scorer.query_count,
            "best_score": float(current_scores.max()),
            "mean_score": float(current_scores.mean()),
        }
    ]
    mutable_count = len(getattr(space, "mutable_positions", tuple(range(space.length))))
    block = int(
        getattr(space, "single_mutant_count", mutable_count * (space.alphabet_size - 1))
    )
    if block == 0:
        return _finalize(
            "ism_greedy",
            archive,
            config,
            scorer,
            {"sweeps": 0, "stop_reason": "fixed_region_exhausted"},
            trajectory,
        )
    while scorer.remaining is None or scorer.remaining >= block:
        before = scorer.query_count
        order = rng.permutation(len(current))
        count = (
            len(order)
            if scorer.remaining is None
            else min(len(order), scorer.remaining // block)
        )
        chosen = order[:count]
        mutants = space.single_mutants(current[chosen])
        batch = _score(scorer, archive, mutants)
        passed = scorer.constraints_passed(mutants)
        current_scores = scorer.current_scores(current)
        for offset, index in enumerate(chosen):
            values = batch.scores[offset * block : (offset + 1) * block]
            valid = passed[offset * block : (offset + 1) * block]
            if not valid.any():
                continue
            best = int(np.flatnonzero(valid)[np.argmax(values[valid])])
            if values[best] > current_scores[index]:
                current[index] = mutants[offset * block + best]
                current_scores[index] = values[best]
        retained = scorer.constraints_passed(current)
        current = current[retained]
        current_scores = scorer.current_scores(current)
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(current_scores.max()),
                "mean_score": float(current_scores.mean()),
            }
        )
        if not len(chosen) or scorer.query_count == before:
            break
    return _finalize(
        "ism_greedy",
        archive,
        config,
        scorer,
        {"sweeps": len(trajectory) - 1},
        trajectory,
    )


@register_algorithm("simulated_annealing")
def simulated_annealing(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    current = initial.copy()
    current_scores = _score(scorer, archive, current).scores
    temperature = max(float(np.std(current_scores)), 1e-3)
    trajectory = []
    while scorer.remaining is None or scorer.remaining >= len(current):
        before = scorer.query_count
        proposals = space.mutate_one(current, rng)
        values = _score(scorer, archive, proposals).scores
        passed = scorer.constraints_passed(proposals)
        current_scores = scorer.current_scores(current)
        current_passed = scorer.constraints_passed(current)
        progress = scorer.query_count / max(config.budget, 1)
        temp = max(temperature * (0.001**progress), 1e-6)
        accept = np.zeros(len(current), dtype=bool)
        accept[passed] = (
            (~current_passed[passed])
            | (values[passed] >= current_scores[passed])
            | (
                rng.random(int(passed.sum()))
                < np.exp(
                    np.minimum((values[passed] - current_scores[passed]) / temp, 0)
                )
            )
        )
        current[accept] = proposals[accept]
        current_scores[accept] = values[accept]
        retained = scorer.constraints_passed(current)
        current = current[retained]
        current_scores = scorer.current_scores(current)
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(current_scores.max()),
                "mean_score": float(current_scores.mean()),
            }
        )
        if scorer.query_count == before:
            break
    return _finalize(
        "simulated_annealing",
        archive,
        config,
        scorer,
        {"initial_temperature": temperature},
        trajectory,
    )


@register_algorithm("genetic_algorithm")
def genetic_algorithm(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    population = initial.copy()
    scores = _score(scorer, archive, population).scores
    population_size = len(population)
    elite_count = max(1, int(round(population_size * config.elite_fraction)))
    trajectory = []
    while scorer.remaining is None or scorer.remaining >= population_size:
        before = scorer.query_count
        retained = scorer.constraints_passed(population)
        population = population[retained]
        if not len(population):
            break
        scores = scorer.current_scores(population)
        elite = population[np.argsort(scores)[-min(elite_count, len(population)) :]]
        if hasattr(space, "offspring"):
            children = space.offspring(
                elite, population_size, config.mutation_rate, rng
            )
        else:
            a = space.encode(elite[rng.integers(len(elite), size=population_size)])
            b = space.encode(elite[rng.integers(len(elite), size=population_size)])
            child = np.where(rng.random(a.shape) < 0.5, a, b)
            mask = rng.random(child.shape) < config.mutation_rate
            child = np.where(
                mask, rng.integers(space.alphabet_size, size=child.shape), child
            )
            children = space.decode(child)
        child_scores = _score(scorer, archive, children).scores
        passed = scorer.constraints_passed(children)
        retained = scorer.constraints_passed(population)
        population = population[retained]
        scores = scorer.current_scores(population)
        combined = np.concatenate([population, children[passed]])
        combined_scores = np.concatenate([scores, child_scores[passed]])
        keep = np.argsort(combined_scores)[-population_size:]
        population, scores = combined[keep], combined_scores[keep]
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(scores.max()),
                "mean_score": float(scores.mean()),
            }
        )
        if scorer.query_count == before:
            break
    return _finalize(
        "genetic_algorithm",
        archive,
        config,
        scorer,
        {"generations": len(trajectory), "elite_count": elite_count},
        trajectory,
    )


@register_algorithm("adalead")
def adalead(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    population = initial.copy()
    scores = _score(scorer, archive, population).scores
    trajectory = []
    while scorer.remaining is None or scorer.remaining >= len(population):
        before = scorer.query_count
        retained = scorer.constraints_passed(population)
        population = population[retained]
        if not len(population):
            break
        scores = scorer.current_scores(population)
        elite_count = max(1, int(round(len(population) * config.elite_fraction)))
        elite = population[np.argsort(scores)[-elite_count:]]
        parents = elite[rng.integers(len(elite), size=len(population))]
        proposals = space.mutate_one(parents, rng)
        values = _score(scorer, archive, proposals).scores
        passed = scorer.constraints_passed(proposals)
        scores = scorer.current_scores(population)
        current_passed = scorer.constraints_passed(population)
        improved = passed & ((~current_passed) | (values > scores))
        population[improved] = proposals[improved]
        scores[improved] = values[improved]
        retained = scorer.constraints_passed(population)
        population = population[retained]
        scores = scorer.current_scores(population)
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(scores.max()),
                "mean_score": float(scores.mean()),
            }
        )
        if scorer.query_count == before:
            break
    return _finalize(
        "adalead", archive, config, scorer, {"steps": len(trajectory)}, trajectory
    )


def _configured_space(length, alphabet, config, algorithm, initial=None):
    if not config.is_CDS_optimized:
        return _sequence_space(length, alphabet)
    if initial is None:
        raise ValueError("initial is required when is_CDS_optimized=True")
    from .codons import SynonymousCodonSpace

    values = normalize_nucleotide_sequences(initial)
    if not len(values):
        raise ValueError("initial must contain at least one CDS")
    space = SynonymousCodonSpace(str(values[0]))
    requested = _sequence_space(length, alphabet)
    if requested.length != space.length or requested.alphabet != "ACGT":
        raise ValueError("length must match the initial CDS and alphabet must be ACGT")
    space.validate(values)
    return space


def _apply_fixed_region(space, fixed_region, initial, rng):
    positions = resolve_fixed_positions(fixed_region, space.length)
    if not positions:
        return space
    if initial is None:
        template = str(space.random(1, rng)[0])
    else:
        values = space.validate(initial)
        if not len(values):
            raise ValueError("initial must contain at least one sequence")
        template = str(values[0])
    restrict = getattr(space, "with_fixed_positions", None)
    if restrict is not None:
        space = restrict(template, positions)
    return FixedRegionSequenceSpace(space, template, positions)


def run(
    name: str,
    scorer: SequenceScorer,
    length: int,
    config: OptimizationConfig,
    *,
    alphabet: str = "ACGT",
    initial: Sequence[str] | np.ndarray | None = None,
    objective: TargetObjective
    | MultiTaskObjective
    | list[str | float]
    | tuple[str | float, ...]
    | None = None,
    constraint: ConstraintSpec | None = None,
    seed: int = 0,
    count_repeated: bool = False,
) -> OptimizationResult:
    """Run one algorithm against a single- or multi-task activity scorer.

    The model output is scalarized by ``objective``. If omitted, output zero is
    maximized, which is the natural default for a single-task model. Direction
    lists use an equal-weight mean across active outputs.
    """
    key = str(name).strip().lower()
    space = _configured_space(length, alphabet, config, key, initial)
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown algorithm {name!r}; choose from {', '.join(available_algorithms())}"
        )
    rng = np.random.default_rng(seed)
    constraint_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0xC0DEC0DE])
    )
    random.seed(seed)
    space = _apply_fixed_region(space, config.fixed_region, initial, rng)
    prepared = _initial(initial, space, config, rng)
    objective_scorer: ObjectiveScorer | None
    if isinstance(scorer, CachedScorer):
        if objective is not None:
            raise ValueError(
                "pass an uncached activity scorer when using a multi-task objective"
            )
        budgeted = scorer
        if constraint is not None:
            budgeted.set_constraint(constraint, constraint_rng)
        if key == "random":
            budgeted.count_repeated = True
        objective_scorer = None
    else:
        objective_scorer = ObjectiveScorer(
            scorer,
            TargetObjective() if objective is None else objective,
        )
        proposal_count_algorithms = {
            "random",
            "ledidi",
            "cbas",
            "vanilla_activation_maximization",
            "fast_seqprop",
            "den",
        }
        budgeted = CachedScorer(
            objective_scorer,
            config.budget,
            count_repeated=count_repeated or key in proposal_count_algorithms,
            constraint=constraint,
            constraint_rng=constraint_rng,
        )
    if budgeted.budget is None:
        budgeted.budget = config.budget
    prepared = _constraint_initial(prepared, budgeted, space, config, rng)
    result = _REGISTRY[key](prepared, budgeted, space, config, rng)
    fixed_positions = tuple(getattr(space, "fixed_positions", ()))
    if fixed_positions:
        result.metadata["fixed_positions"] = list(fixed_positions)
    if objective_scorer is not None:
        activities = objective_scorer.activities_for(result.sequences)
        result.attach_activities(
            activities,
            objective_scorer.resolved_task_names(activities.shape[1]),
        )
        result.metadata["objective"] = objective_scorer.objective.describe(
            activities.shape[1], objective_scorer.task_names
        )
        if isinstance(objective_scorer.objective, MultiTaskObjective):
            result.metadata["objective_evaluation"] = (
                objective_scorer.objective.evaluate(
                    activities, objective_scorer.task_names
                )
            )
    if budgeted.constraint_evaluator is not None:
        evaluator = budgeted.constraint_evaluator
        result.attach_constraint_scores(
            evaluator.probabilities_for(result.sequences), evaluator.names
        )
        result.metadata["constraint"] = {
            **evaluator.describe(),
            "screened_query_count": int(evaluator.evaluated_counts[0]),
            "counted_query_count": result.query_count,
            "model_query_count": budgeted.model_query_count,
        }
    return result


def run_many(
    names: Iterable[str],
    scorer: SequenceScorer,
    length: int,
    config: OptimizationConfig,
    *,
    alphabet: str = "ACGT",
    initial: Sequence[str] | np.ndarray | None = None,
    objective: TargetObjective
    | MultiTaskObjective
    | list[str | float]
    | tuple[str | float, ...]
    | None = None,
    constraint: ConstraintSpec | None = None,
    seed: int = 0,
) -> dict[str, OptimizationResult]:
    """Run algorithms independently; each receives a fresh budget/cache."""
    return {
        str(name): run(
            name,
            scorer,
            length,
            config,
            alphabet=alphabet,
            initial=initial,
            objective=objective,
            constraint=constraint,
            seed=seed + offset,
        )
        for offset, name in enumerate(names)
    }


def optimize(
    model: SequenceScorer,
    length: int,
    initial: Sequence[str] | np.ndarray | None,
    objective: TargetObjective
    | MultiTaskObjective
    | list[str | float]
    | tuple[str | float, ...]
    | None,
    num_candidates: int,
    *,
    algorithm: str = "genetic_algorithm",
    budget: int | None = None,
    saturation: SaturationConfig | None = None,
    diversity: int = 5,
    fixed_region: Sequence[int | str | slice] | None = None,
    alphabet: str = "ACGT",
    constraint: ConstraintSpec | None = None,
    seed: int = 0,
    mutation_rate: float = 0.02,
    elite_fraction: float = 0.2,
    learning_rate: float = 0.1,
    generator_learning_rate: float = 3e-3,
    generator_pretrain_epochs: int = 20,
    generator_update_epochs: int = 5,
    cbas_quantile: float = 0.8,
    cbas_batch_size: int = 256,
    den_seed_pretrain_epochs: int = 100,
    den_diversity_weight: float = 0.1,
    ledidi_edit_penalty: float = 0.1,
    ledidi_temperature: float = 1.0,
    ledidi_epsilon: float = 1e-4,
    basesearch_decay: float = 0.995,
    basesearch_queue_size: int = 1024,
    basesearch_max_stalled: int = 1024,
    is_CDS_optimized: bool = False,
    count_repeated: bool = False,
) -> (
    OptimizationResult | SaturationResult | list[OptimizationResult | SaturationResult]
):
    """Optimize sequences with a parameterized model.

    ``diversity`` is the minimum pairwise Hamming distance among returned
    sequences. ``fixed_region`` holds zero-based positions or Python-style
    slices to one shared sequence template. Numeric objective entries are
    raw-output lower targets.
    """
    parameters = dict(
        num_candidates=num_candidates,
        diversity=diversity,
        fixed_region=fixed_region,
        mutation_rate=mutation_rate,
        elite_fraction=elite_fraction,
        learning_rate=learning_rate,
        generator_learning_rate=generator_learning_rate,
        generator_pretrain_epochs=generator_pretrain_epochs,
        generator_update_epochs=generator_update_epochs,
        cbas_quantile=cbas_quantile,
        cbas_batch_size=cbas_batch_size,
        den_seed_pretrain_epochs=den_seed_pretrain_epochs,
        den_diversity_weight=den_diversity_weight,
        ledidi_edit_penalty=ledidi_edit_penalty,
        ledidi_temperature=ledidi_temperature,
        ledidi_epsilon=ledidi_epsilon,
        basesearch_decay=basesearch_decay,
        basesearch_queue_size=basesearch_queue_size,
        basesearch_max_stalled=basesearch_max_stalled,
        is_CDS_optimized=is_CDS_optimized,
    )
    algorithm_key = str(algorithm).strip().lower()
    if algorithm_key == "random":
        if saturation is not None:
            raise ValueError(
                "saturation is not supported by random search; set budget instead"
            )
        if budget is None:
            budget = _DEFAULT_BUDGET
        count_repeated = True
    automatic_saturation = budget is None and saturation is None
    if automatic_saturation:
        saturation = SaturationConfig()
    if saturation is not None and budget is not None:
        raise ValueError("budget is invalid when saturation is set; omit budget")
    if saturation is not None and not isinstance(saturation, SaturationConfig):
        raise TypeError("saturation must be a SaturationConfig or None")

    def execute(one_initial, one_seed):
        if saturation is not None:
            from .saturation import _optimize_with_saturation

            saturated = _optimize_with_saturation(
                model,
                length,
                algorithm=algorithm,
                alphabet=alphabet,
                initial=one_initial,
                objective=objective,
                constraint=constraint,
                saturation=saturation,
                seed=one_seed,
                count_repeated=count_repeated or automatic_saturation,
                **parameters,
            )
            return saturated[0.99] if automatic_saturation else saturated
        resolved_budget = _DEFAULT_BUDGET if budget is None else budget
        config = OptimizationConfig(budget=resolved_budget, **parameters)
        return run(
            algorithm,
            model,
            length,
            config,
            alphabet=alphabet,
            initial=one_initial,
            objective=objective,
            constraint=constraint,
            seed=one_seed,
            count_repeated=count_repeated,
        )

    if not is_CDS_optimized:
        return execute(initial, seed)
    if initial is None:
        raise ValueError("initial is required when is_CDS_optimized=True")
    cds_values = normalize_nucleotide_sequences(initial)
    if not len(cds_values):
        raise ValueError("initial must contain at least one CDS")
    from .codons import SynonymousCodonSpace

    prepared_cds = []
    seed_config = OptimizationConfig(
        budget=max(_DEFAULT_BUDGET, num_candidates) if budget is None else budget,
        **parameters,
    )
    for offset, value in enumerate(cds_values):
        reference = str(value)
        space = SynonymousCodonSpace(reference)
        if space.length != length:
            raise ValueError("length must match every initial CDS")
        local_seed = seed + offset
        local_rng = np.random.default_rng(local_seed)
        space = _apply_fixed_region(space, fixed_region, [reference], local_rng)
        population = _initial(
            None,
            space,
            seed_config,
            local_rng,
            anchor=reference,
        )
        prepared_cds.append((reference, population, local_seed))

    def optimize_cds(reference, population, local_seed):
        result = execute(population, local_seed)
        selections = (
            result.selections.values()
            if isinstance(result, SaturationResult)
            else (result,)
        )
        for selection in selections:
            selection.metadata["initial_cds"] = reference
        return result

    results = [optimize_cds(*values) for values in prepared_cds]
    return results[0] if len(results) == 1 else results


def _named_optimizer(name, public_name=None, doc=None, *, default_budget=None):
    def function(model, length, initial, objective, num_candidates, **kwargs):
        if "algorithm" in kwargs:
            raise TypeError(f"{function.__name__}() does not accept algorithm")
        return optimize(
            model,
            length,
            initial,
            objective,
            num_candidates,
            algorithm=name,
            **kwargs,
        )

    function.__name__ = public_name or f"{name}_optimize"
    function.__qualname__ = function.__name__
    function.__module__ = __name__
    function.__doc__ = doc or f"Optimize sequences with {name}."
    parameters = [
        parameter
        for parameter in inspect.signature(optimize).parameters.values()
        if parameter.name != "algorithm"
    ]
    if default_budget is not None:
        parameters = [
            parameter.replace(default=default_budget)
            if parameter.name == "budget"
            else parameter
            for parameter in parameters
        ]
    function.__signature__ = inspect.signature(optimize).replace(parameters=parameters)
    return function


basesearch_optimize = _named_optimizer(
    "basesearch",
    "basesearch_optimize",
    "Priority-decayed black-box search with unique mutation proposals.",
)
random_optimize = _named_optimizer(
    "random",
    "random_optimize",
    "Uniform random candidate sampling with a fixed proposal budget.",
    default_budget=_DEFAULT_BUDGET,
)
ism_optimize = _named_optimizer("ism_greedy", "ism_optimize")
simulated_annealing_optimize = _named_optimizer(
    "simulated_annealing", "simulated_annealing_optimize"
)
genetic_algorithm_optimize = _named_optimizer(
    "genetic_algorithm", "genetic_algorithm_optimize"
)
adalead_optimize = _named_optimizer("adalead", "adalead_optimize")
beam_search_optimize = _named_optimizer("beam_search", "beam_search_optimize")
cross_entropy_optimize = _named_optimizer(
    "cross_entropy_method", "cross_entropy_optimize"
)
cbas_optimize = _named_optimizer("cbas", "cbas_optimize")
vanilla_activation_maximization_optimize = _named_optimizer(
    "vanilla_activation_maximization", "vanilla_activation_maximization_optimize"
)
fast_seqprop_optimize = _named_optimizer("fast_seqprop", "fast_seqprop_optimize")
ledidi_optimize = _named_optimizer(
    "ledidi",
    "ledidi_optimize",
    "Ledidi-style single-output maximization with an edit penalty.",
)
den_optimize = _named_optimizer("den", "den_optimize")


# Requested spelling retained as a public alias; lower-case ``ism_optimize`` is
# preferred in Python code.
ISM_optimize = ism_optimize


# Import optional advanced implementations after the registry and common
# helpers are defined. Importing this module registers their public names.
from . import advanced as _advanced  # noqa: E402,F401
