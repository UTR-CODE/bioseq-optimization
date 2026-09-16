"""Run selected optimizers and combine their retained candidates."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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

DIFFERENTIABLE_ALGORITHMS = frozenset(
    {
        "den",
        "fast_seqprop",
        "ledidi",
        "vanilla_activation_maximization",
    }
)


class _SharedScorer:
    def __init__(self, scorer: SequenceScorer) -> None:
        self.scorer = scorer
        self.model_query_count = 0
        self.model_batch_sizes: list[int] = []
        self.soft_query_count = 0
        self.soft_batch_sizes: list[int] = []
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
            self.model_query_count += len(missing)
            configured_batch_size = int(
                getattr(self.scorer, "batch_size", len(missing))
            )
            self.model_batch_sizes.append(min(len(missing), configured_batch_size))
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
        batch_size = len(probabilities)
        self.soft_query_count += batch_size
        self.soft_batch_sizes.append(batch_size)
        return self.scorer.score_soft(probabilities)


@dataclass
class OptimizerSuiteResult:
    """Results and timing for a group of independently executed optimizers."""

    results: dict[str, OptimizationResult]
    elapsed_seconds: dict[str, float]
    model_query_counts: dict[str, int]
    max_model_batch_sizes: dict[str, int]
    soft_query_counts: dict[str, int]
    max_soft_batch_sizes: dict[str, int]
    initial_sequences: np.ndarray
    initial_count: int

    def to_frame(self):
        """Return every retained candidate in one rank-annotated table."""
        import pandas as pd

        frames = []
        for algorithm, result in self.results.items():
            frame = result.to_frame().copy()
            frame.insert(0, "rank", np.arange(1, len(frame) + 1))
            frame.insert(0, "algorithm", algorithm)
            frame["query_count"] = result.query_count
            frame["model_query_count"] = self.model_query_counts[algorithm]
            frame["max_discrete_batch_size"] = self.max_model_batch_sizes[algorithm]
            frame["soft_query_count"] = self.soft_query_counts[algorithm]
            frame["max_soft_batch_size"] = self.max_soft_batch_sizes[algorithm]
            frame["elapsed_seconds"] = self.elapsed_seconds[algorithm]
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def summary_frame(self):
        """Return one performance and score summary row per optimizer."""
        import pandas as pd

        rows = []
        for algorithm, result in self.results.items():
            elapsed = self.elapsed_seconds[algorithm]
            rows.append(
                {
                    "algorithm": algorithm,
                    "initial_count": self.initial_count,
                    "candidate_count": len(result.sequences),
                    "query_count": result.query_count,
                    "model_query_count": self.model_query_counts[algorithm],
                    "max_discrete_batch_size": self.max_model_batch_sizes[algorithm],
                    "soft_query_count": self.soft_query_counts[algorithm],
                    "max_soft_batch_size": self.max_soft_batch_sizes[algorithm],
                    "elapsed_seconds": elapsed,
                    "queries_per_second": result.query_count / max(elapsed, 1e-12),
                    "best_score": float(result.scores.max()),
                    "mean_score": float(result.scores.mean()),
                    "stop_reason": result.metadata.get("stop_reason", "budget"),
                }
            )
        return pd.DataFrame(rows)

    def save_csv(self, output_dir: str | Path) -> dict[str, Path]:
        """Write combined, summary, and per-optimizer CSV files."""
        import pandas as pd

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "candidates": directory / "candidates.csv",
            "summary": directory / "summary.csv",
            "initial": directory / "initial.csv",
        }
        self.to_frame().to_csv(paths["candidates"], index=False)
        self.summary_frame().to_csv(paths["summary"], index=False)
        pd.DataFrame({"sequence": self.initial_sequences}).to_csv(
            paths["initial"], index=False
        )
        for algorithm, result in self.results.items():
            path = directory / f"{algorithm}.csv"
            result.to_frame().to_csv(path, index=False)
            paths[algorithm] = path
        return paths


def optimize_algorithms(
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
    budget: int | None = None,
    differentiable_model: SequenceScorer | None = None,
    alphabet: str = "ACGT",
    constraint: ConstraintSpec | None = None,
    seed: int = 0,
    optimizer_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    share_model_cache: bool = True,
    **common_kwargs: Any,
) -> OptimizerSuiteResult:
    """Run selected algorithms independently from one shared initial population.

    Black-box algorithms use ``model``. Gradient algorithms use
    ``differentiable_model`` when supplied, or ``model`` when it implements
    ``score_soft``. ``optimizer_kwargs`` overrides common options for a named
    algorithm.
    """
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
    if budget < len(initial_values):
        raise ValueError(
            f"budget must be at least the initial population size {len(initial_values)}"
        )
    if not isinstance(share_model_cache, bool):
        raise TypeError("share_model_cache must be a boolean")
    per_algorithm = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    unused = sorted(set(per_algorithm) - set(names))
    if unused:
        raise KeyError(f"optimizer_kwargs contains unselected algorithms: {unused}")

    needs_differentiable = sorted(set(names) & DIFFERENTIABLE_ALGORITHMS)
    gradient_model = differentiable_model
    if gradient_model is None and hasattr(model, "score_soft"):
        gradient_model = model
    if needs_differentiable and gradient_model is None:
        raise TypeError(
            "differentiable_model is required for gradient algorithms: "
            + ", ".join(needs_differentiable)
        )

    black_box_scorer = _SharedScorer(model) if share_model_cache else None
    gradient_scorer = (
        _SharedScorer(gradient_model)
        if share_model_cache and gradient_model is not None
        else None
    )

    shared_options = dict(common_kwargs)
    inference_batch_size = int(
        getattr(model, "batch_size", min(max(len(initial_values), 256), 4096))
    )
    population_batch_size = min(len(initial_values), inference_batch_size)
    shared_options.setdefault("basesearch_batch_size", population_batch_size)
    shared_options.setdefault("cbas_batch_size", population_batch_size)
    shared_options.setdefault("cross_entropy_batch_size", population_batch_size)
    options_by_name: dict[str, dict[str, Any]] = {}
    reserved_names = {
        "algorithm",
        "alphabet",
        "budget",
        "constraint",
        "initial",
        "length",
        "model",
        "num_candidates",
        "objective",
        "seed",
    }
    for name in names:
        options = {**shared_options, **dict(per_algorithm.get(name, {}))}
        reserved = sorted(set(options) & reserved_names)
        if reserved:
            raise TypeError(
                "optimizer options cannot override suite arguments: "
                + ", ".join(reserved)
            )
        options_by_name[name] = options

    results: dict[str, OptimizationResult] = {}
    elapsed_seconds: dict[str, float] = {}
    model_query_counts: dict[str, int] = {}
    max_model_batch_sizes: dict[str, int] = {}
    soft_query_counts: dict[str, int] = {}
    max_soft_batch_sizes: dict[str, int] = {}
    for offset, name in enumerate(names):
        if name in DIFFERENTIABLE_ALGORITHMS:
            scorer = gradient_scorer or _SharedScorer(gradient_model)
        else:
            scorer = black_box_scorer or _SharedScorer(model)
        model_queries_before = getattr(scorer, "model_query_count", 0)
        batch_index = len(getattr(scorer, "model_batch_sizes", ()))
        soft_queries_before = getattr(scorer, "soft_query_count", 0)
        soft_batch_index = len(getattr(scorer, "soft_batch_sizes", ()))
        started = time.perf_counter()
        result = optimize(
            scorer,
            length,
            initial_values,
            objective,
            num_candidates,
            algorithm=name,
            budget=budget,
            alphabet=alphabet,
            constraint=constraint,
            seed=seed + offset,
            **options_by_name[name],
        )
        if not isinstance(result, OptimizationResult):
            raise TypeError(
                "optimize_algorithms requires fixed-budget optimizer results"
            )
        elapsed_seconds[name] = time.perf_counter() - started
        model_queries_after = getattr(scorer, "model_query_count", result.query_count)
        model_query_counts[name] = int(model_queries_after - model_queries_before)
        current_batch_sizes = getattr(scorer, "model_batch_sizes", ())[batch_index:]
        max_model_batch_sizes[name] = max(current_batch_sizes, default=0)
        soft_queries_after = getattr(scorer, "soft_query_count", soft_queries_before)
        soft_query_counts[name] = int(soft_queries_after - soft_queries_before)
        current_soft_batch_sizes = getattr(scorer, "soft_batch_sizes", ())[
            soft_batch_index:
        ]
        max_soft_batch_sizes[name] = max(current_soft_batch_sizes, default=0)
        result.metadata["suite_model_query_count"] = model_query_counts[name]
        result.metadata["suite_max_model_batch_size"] = max_model_batch_sizes[name]
        result.metadata["suite_soft_query_count"] = soft_query_counts[name]
        result.metadata["suite_max_soft_batch_size"] = max_soft_batch_sizes[name]
        results[name] = result
    return OptimizerSuiteResult(
        results,
        elapsed_seconds,
        model_query_counts,
        max_model_batch_sizes,
        soft_query_counts,
        max_soft_batch_sizes,
        initial_values.copy(),
        len(initial_values),
    )
