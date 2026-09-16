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
from .saturation import SaturationConfig

DIFFERENTIABLE_ALGORITHMS = frozenset(
    {
        "den",
        "fast_seqprop",
        "ledidi",
        "vanilla_activation_maximization",
    }
)


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
    alphabet: str = "ACGT",
    constraint: ConstraintSpec | None = None,
    seed: int = 0,
    **common_kwargs: Any,
):
    """Run all 13 algorithms and return unified candidate pool as pandas DataFrame.

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
        alphabet: DNA/RNA alphabet (default: "ACGT")
        constraint: Sequence constraints (optional)
        seed: Random seed for reproducibility
        **common_kwargs: Additional options passed to all algorithms

    Returns:
        pandas DataFrame with columns: sequence, algorithm, score, activity_*

    Notes:
        - Random search: generates 100,000 sequences, returns top 100
        - Other algorithms: auto-stop at saturation plateau (0.9)
        - Diversity control: ensures minimum Hamming distance of 5 within each algorithm
    """
    import pandas as pd

    # Default to all 13 algorithms
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

    initial_values = normalize_nucleotide_sequences(initial)
    if len(initial_values) < num_candidates:
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

    # Set up gradient model for differentiable algorithms
    needs_differentiable = sorted(set(names) & DIFFERENTIABLE_ALGORITHMS)
    gradient_model = differentiable_model
    if gradient_model is None and hasattr(model, "score_soft"):
        gradient_model = model
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
        if "budget" in algo_options or "saturation" in algo_options:
            pass
        elif name == "random":
            # Random search: generate 100,000 sequences, return top 100
            algo_options["budget"] = 100_000
        else:
            # Other algorithms: stop at the 0.9 saturation plateau
            algo_options["saturation"] = SaturationConfig(
                fractions=(0.9,),
                minimum_queries=len(initial_values),
                plateau_patience_queries=len(initial_values) * 2,
                maximum_queries=100_000,
            )

        # Run optimization
        result = optimize(
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

