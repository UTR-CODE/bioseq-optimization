"""Run selected optimizers and combine their retained candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .algorithms import available_algorithms, optimize
from .api import (
    OptimizationResult,
    ScoreBatch,
    SequenceScorer,
    as_score_batch,
    normalize_nucleotide_sequences,
)
from .constraints import ConstraintSpec
from .objectives import MultiTaskObjective, TargetObjective
from .saturation import SaturationConfig, SaturationResult

DIFFERENTIABLE_ALGORITHMS = frozenset(
    {
        "den",
        "fast_seqprop",
        "ledidi",
        "vanilla_activation_maximization",
    }
)


def _supports_soft_scoring(model: SequenceScorer | None) -> bool:
    if model is None or not callable(getattr(model, "score_soft", None)):
        return False
    return bool(getattr(model, "supports_soft_scoring", True))


class _TaskNamedScorer:
    def __init__(self, scorer: SequenceScorer, task_names: Sequence[str]) -> None:
        self.scorer = scorer
        self.task_names = tuple(str(name).strip() for name in task_names)

    def __getattr__(self, name):
        return getattr(self.scorer, name)

    def score(self, sequences):
        return self.scorer.score(sequences)

    def score_soft(self, probabilities):
        return self.scorer.score_soft(probabilities)


class _SharedScorer:
    """Shared model scorer with caching across algorithms."""

    def __init__(self, scorer: SequenceScorer) -> None:
        self.scorer = scorer
        self._cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    def __getattr__(self, name: str):
        return getattr(self.scorer, name)

    def score(self, sequences) -> ScoreBatch:
        values = normalize_nucleotide_sequences(sequences)
        missing = list(
            dict.fromkeys(str(value) for value in values if value not in self._cache)
        )
        if missing:
            batch = as_score_batch(self.scorer.score(np.asarray(missing, dtype=str)))
            if len(batch.scores) != len(missing):
                raise ValueError(
                    f"scorer returned {len(batch.scores)} rows for {len(missing)} sequences"
                )
            for index, sequence in enumerate(missing):
                uncertainty = (
                    None
                    if batch.uncertainties is None
                    else np.asarray(batch.uncertainties[index]).copy()
                )
                self._cache[sequence] = (
                    np.asarray(batch.scores[index]).copy(),
                    uncertainty,
                )
        scores = np.stack([self._cache[str(value)][0] for value in values])
        cached_uncertainties = [self._cache[str(value)][1] for value in values]
        uncertainties = (
            None
            if all(value is None for value in cached_uncertainties)
            else np.stack(
                [
                    np.zeros_like(score) if uncertainty is None else uncertainty
                    for score, uncertainty in zip(scores, cached_uncertainties)
                ]
            )
        )
        return ScoreBatch(scores, uncertainties)

    def score_soft(self, probabilities):
        return self.scorer.score_soft(probabilities)


def generate_designs(
    model: SequenceScorer,
    length: int,
    initial: Sequence[str] | np.ndarray,
    objective: TargetObjective
    | MultiTaskObjective
    | list[str | float]
    | tuple[str | float, ...]
    | None,
    *,
    algorithms: Sequence[str] | None = None,
    num_candidates: int = 100,
    diversity: int = 5,
    weights: Sequence[float] | None = None,
    differentiable_model: SequenceScorer | None = None,
    tasks: Sequence[str] | None = None,
    alphabet: str = "ACGT",
    constraint: ConstraintSpec | None = None,
    seed: int = 0,
    budget: int | None = None,
    saturation: SaturationConfig | None = None,
    fixed_region: Sequence[int | str | slice] | None = None,
    is_CDS_optimized: bool = False,
    count_repeated: bool = False,
    **common_kwargs: Any,
):
    """Run all 13 optimization algorithms and return one candidate DataFrame.

    Args:
        model: Predictor model for sequence scoring
        length: Sequence length to optimize
        initial: Initial population sequences
        objective: Optimization objective (e.g., ["maximize"], ["minimize", "maximize"])
        algorithms: Algorithm names to run (default: all 13)
        num_candidates: Candidates per algorithm (default: 100)
        diversity: Minimum Hamming distance between candidates (default: 5)
        weights: Relative importance per model output (default: equal weighting)
        differentiable_model: Model for gradient algorithms (optional)
        tasks: Optional output names for this generate_designs call
        alphabet: DNA/RNA alphabet (default: "ACGT")
        constraint: Sequence constraints (optional)
        seed: Random seed for reproducibility
        budget: Fixed query budget per algorithm; ``None`` uses saturation
            stopping (random search uses its finite default budget)
        saturation: Saturation rules for non-random algorithms; cannot be
            combined with ``budget`` or used with random search
        fixed_region: Positions or slices held unchanged during optimization
        is_CDS_optimized: Restrict candidates to synonymous CDS substitutions
        count_repeated: Count repeated proposals against the query budget
        **common_kwargs: Algorithm-specific hyperparameters passed to every algorithm

    Returns:
        pandas DataFrame with columns: sequence, algorithm, score, activity_*

    Notes:
        - Random search: uses a finite 100,000-query default, returns top 100
        - Other algorithms: auto-stop at plateau, return diverse relaxed-knee sets
          with population-scaled minimum/patience windows (10N/40N), a
          three-proposal-batch patience floor, and no implicit query ceiling
        - Diversity control: ensures minimum Hamming distance of 5 within each algorithm
    """
    # The suite is the complete comparison unless the caller selects a subset.
    if algorithms is None:
        algorithms = tuple(available_algorithms())

    if isinstance(algorithms, (str, bytes)):
        raise TypeError("algorithms must be a sequence of algorithm names")
    names = tuple(str(name).strip().lower() for name in algorithms)
    if not names:
        raise ValueError("algorithms must contain at least one name")
    if len(set(names)) != len(names):
        raise ValueError("algorithms must not contain duplicates")
    unknown = sorted(set(names) - set(available_algorithms()))
    if unknown:
        raise KeyError(
            f"unknown algorithms {unknown}; choose from {', '.join(available_algorithms())}"
        )
    if budget is not None and saturation is not None:
        raise ValueError("budget and saturation cannot be used together")
    if saturation is not None and "random" in names:
        raise ValueError("saturation is not supported by random search; set budget instead")

    initial_values = normalize_nucleotide_sequences(initial)
    if not len(initial_values):
        raise ValueError("initial must contain at least one sequence")
    if not is_CDS_optimized and len(initial_values) < num_candidates:
        raise ValueError(
            f"initial must contain at least {num_candidates} sequences; "
            f"received {len(initial_values)}"
        )

    # Objective weights: attach to the objective so every algorithm shares them.
    if weights is not None:
        if isinstance(objective, (list, tuple)):
            objective = MultiTaskObjective(directions=objective, weights=weights)
        elif isinstance(objective, MultiTaskObjective):
            objective = replace(objective, weights=weights)
        else:
            raise TypeError(
                "weights require an objective given as a directions list or "
                "MultiTaskObjective"
            )

    if tasks is not None:
        task_names = tuple(str(name).strip() for name in tasks)
        if not task_names or any(not name for name in task_names):
            raise ValueError("tasks must contain non-empty names")
        if len(set(task_names)) != len(task_names):
            raise ValueError("tasks must contain unique names")
        model = _TaskNamedScorer(model, task_names)
        if differentiable_model is not None:
            differentiable_model = _TaskNamedScorer(differentiable_model, task_names)

    gradient_model = differentiable_model
    if gradient_model is None and _supports_soft_scoring(model):
        gradient_model = model
    if gradient_model is not None and not _supports_soft_scoring(gradient_model):
        raise TypeError("differentiable_model must implement score_soft(probabilities)")

    # Set up gradient model for differentiable algorithms
    needs_differentiable = sorted(set(names) & DIFFERENTIABLE_ALGORITHMS)
    if needs_differentiable and gradient_model is None:
        raise TypeError(
            "differentiable_model is required for gradient algorithms: "
            + ", ".join(needs_differentiable)
        )

    # Shared model caching across algorithms
    black_box_scorer = _SharedScorer(model)
    gradient_scorer = _SharedScorer(gradient_model) if gradient_model is not None else None

    # Common options for all algorithms
    shared_options = dict(common_kwargs)
    shared_options["diversity"] = diversity
    shared_options["fixed_region"] = fixed_region
    shared_options["is_CDS_optimized"] = is_CDS_optimized
    shared_options["count_repeated"] = count_repeated
    if budget is not None:
        shared_options["budget"] = budget
    if saturation is not None:
        shared_options["saturation"] = saturation

    # Collect results from all algorithms
    all_rows = []

    for offset, name in enumerate(names):
        # Select appropriate scorer
        if name in DIFFERENTIABLE_ALGORITHMS:
            scorer = gradient_scorer
        else:
            scorer = black_box_scorer

        # Set algorithm-specific options
        algo_options = dict(shared_options)

        # An explicit budget or saturation config overrides the suite defaults.
        if budget is not None or saturation is not None:
            pass
        elif name == "random":
            # Random search cannot detect saturation; use its finite default.
            algo_options["budget"] = 100_000
        else:
            # No implicit ceiling: stop on plateau and use relaxed-knee selection.
            algo_options["saturation"] = SaturationConfig(fractions=(0.9,))

        # Run optimization
        raw_result = optimize(
            scorer,
            length,
            initial_values,
            objective,
            num_candidates,
            algorithm=name,
            alphabet=alphabet,
            constraint=constraint,
            seed=seed + offset,
            **algo_options,
        )

        results = raw_result if isinstance(raw_result, list) else [raw_result]
        for result in results:
            if isinstance(result, SaturationResult):
                result = result.selected
            if not isinstance(result, OptimizationResult):
                raise TypeError(
                    "generate_designs requires fixed-budget optimizer results"
                )

            # Extract data from result
            for i, sequence in enumerate(result.sequences):
                row = {
                    "sequence": sequence,
                    "algorithm": name,
                    "score": float(result.scores[i]),
                }

                # Add activity columns if available
                if result.activities is not None and result.task_names:
                    for j, task_name in enumerate(result.task_names):
                        row[f"activity_{task_name}"] = float(result.activities[i, j])

                all_rows.append(row)

    # Convert to DataFrame
    return pd.DataFrame(all_rows)
