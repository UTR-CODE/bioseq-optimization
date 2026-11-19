"""Run selected optimizers and combine their retained candidates."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace, asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import sys
from threading import RLock
import time
from typing import Any

import numpy as np
import pandas as pd

from .algorithms import available_algorithms, optimize
from .api import (
    OptimizationResult,
    ScoreBatch,
    SequenceScorer,
    SequencePredictor,
    as_score_batch,
    normalize_nucleotide_sequences,
    resolve_fixed_positions,
)
from .constraints import ConstraintSpec
from .objectives import CustomObjective, MultiTaskObjective, TargetObjective
from .saturation import SaturationConfig, SaturationResult, resolve_search_config
from .exploration import DiversitySearchConfig, MotifPositionDescriptor

DIFFERENTIABLE_ALGORITHMS = frozenset(
    {
        "den",
        "fast_seqprop",
        "ledidi",
        "vanilla_activation_maximization",
    }
)


_TORCH_SCORING_LOCK = RLock()


@contextmanager
def _reproducible_scoring():
    import torch

    # The backend flags are process-global, including when a caller uses threads.
    with _TORCH_SCORING_LOCK:
        tf32 = torch.backends.cuda.matmul.allow_tf32
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=warn_only or not deterministic)
            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=False, deterministic=True, allow_tf32=False,
            ):
                yield
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cuda.matmul.allow_tf32 = tf32


class _DesignProgress:
    def __init__(self, names):
        self.names = names
        self.completed = 0
        self.started = time.monotonic()
        self.next_refresh = self.started

    def report(self, round_number=None, mean=None):
        now = time.monotonic()
        if round_number is not None and now < self.next_refresh:
            return
        self.next_refresh = now + 5
        n = len(self.names)
        filled = 20 * self.completed // n
        bar = "#" * filled + "-" * (20 - filled)
        status = f"[{bar}] {self.completed}/{n} optimizers complete | {now - self.started:.0f} s"
        if self.completed < n:
            status += f" | {self.names[self.completed]}"
            if round_number is not None:
                status += f" | round {round_number} | archive mean {mean:.4f}"
        print(status, file=sys.stderr, flush=True)


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
        self.report_progress = None

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
    initial: Sequence[str] | np.ndarray | None = None,
    objective: TargetObjective
    | MultiTaskObjective
    | list[str]
    | tuple[str, ...]
    | None = None,
    *,
    output_dir: str | Path | None = None,
    device: str | None = None,
    show_progress: bool = True,
    algorithms: Sequence[str] | None = None,
    num_candidates: int = 100,
    diversity: int = 5,
    weights: Sequence[float] | None = None,
    stop_values: Sequence[float | tuple[float | None, float | None] | None] | None = None,
    stop_tolerance: float | None = 0.5,
    differentiable_model: SequenceScorer | None = None,
    tasks: Sequence[str] | None = None,
    alphabet: str = "ACGT",
    constraint: ConstraintSpec | None = None,
    seed: int = 42,
    gain_fraction: float | None = None,
    gain_fractions: Sequence[float] | None = None,
    score_threshold: float | None = None,
    saturation: SaturationConfig | None = None,
    fixed_region: Sequence[int | str | slice] | None = None,
    is_CDS_optimized: bool = False,
    count_repeated: bool = True,
    exploration: DiversitySearchConfig | None = None,
    descriptor: MotifPositionDescriptor | None = None,
    **common_kwargs: Any,
):
    """Generate candidates with eleven optimizers and retain algorithm provenance.

    Args:
        model: Predictor model for sequence scoring
        length: Sequence length to optimize
        initial: Starting sequences; None generates num_candidates random seeds
        objective: Optimization objective (e.g., ["maximize"], ["minimize", "maximize"])
        show_progress: Show the active optimizer and candidate-archive mean
        algorithms: Algorithm names to run (default: eleven non-random optimizers)
        num_candidates: Candidates per algorithm (default: 100)
        diversity: Minimum Hamming distance between candidates (default: 5)
        weights: Relative importance per model output (default: equal weighting)
        stop_values: Per-output directional caps or (lower, upper) bounds; None means unlimited.
        differentiable_model: Model for gradient algorithms (optional)
        tasks: Optional output names for this generate_designs call
        alphabet: DNA/RNA alphabet (default: "ACGT")
        constraint: Sequence constraints (optional)
        seed: Random seed for reproducibility
        gain_fraction: Observed gain fraction to retain at plateau (default 0.9).
        gain_fractions: Gain stages to return from the same plateau search.
        score_threshold: Stop when a feasible candidate reaches this scalar objective score.
        saturation: Advanced plateau settings for one or more gain fractions.
        fixed_region: Positions or slices held unchanged during optimization
        is_CDS_optimized: Restrict candidates to synonymous CDS substitutions
        count_repeated: Count repeated proposals in query and plateau diagnostics
        **common_kwargs: Algorithm-specific hyperparameters passed to every algorithm

    Returns:
        pandas DataFrame with columns: sequence, algorithm, score, activity_*

    Notes:
        - Gain mode searches to plateau and retains the specified gain stage.
        - Threshold mode retains the archive at the target or reports a plateau.
        - Diversity control: ensures minimum Hamming distance of 5 within each algorithm
    """
    import torch
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")
    removed = sorted(
        {"parallel", "max_batch_size", "max_soft_batch_size", "batch_wait_ms"}
        & common_kwargs.keys()
    )
    if removed:
        raise TypeError(f"Remove {', '.join(removed)}; optimizers now run sequentially")
    if isinstance(model, torch.nn.Module) and not callable(getattr(model, "score", None)):
        tensor = next(model.parameters(), None)
        if tensor is None:
            tensor = next(model.buffers(), None)
        inferred_device = tensor.device if tensor is not None else "cpu"
        model = SequencePredictor(model, tasks=tasks, device=device or inferred_device)
    if initial is None:
        from .seeds import random_seed_sequences
        initial = random_seed_sequences(length, num_candidates, seed=seed)
    if objective is None:
        objective = TargetObjective()
    if output_dir is not None:
        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Use an empty design output directory: {output_dir}")
    if "budget" in common_kwargs:
        raise TypeError("Fixed-budget optimization is unavailable; use score_threshold or gain_fraction")
    if exploration is not None or descriptor is not None:
        raise ValueError("Use score_threshold or gain_fraction for candidate generation")
    resolved = resolve_search_config(saturation, gain_fraction, score_threshold, gain_fractions)
    multiple_stages = len(resolved.fractions) > 1
    if algorithms is None:
        algorithms = tuple(n for n in available_algorithms() if n != "random")

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
    gradient_scorer = (black_box_scorer if gradient_model is model else
                       _SharedScorer(gradient_model) if gradient_model is not None else None)

    # Common options for all algorithms
    shared_options = dict(common_kwargs)
    shared_options["stop_values"] = stop_values
    shared_options["stop_tolerance"] = stop_tolerance
    shared_options["diversity"] = diversity
    shared_options["fixed_region"] = fixed_region
    shared_options["is_CDS_optimized"] = is_CDS_optimized
    shared_options["count_repeated"] = count_repeated
    if score_threshold is not None:
        shared_options["score_threshold"] = score_threshold
    else:
        shared_options["saturation"] = resolved

    all_rows, timings, stage_timings = [], [], []
    result_task_names = ()
    suite_started = time.perf_counter()
    progress = _DesignProgress(names) if show_progress else None
    completed = []
    with _reproducible_scoring():
        for offset, name in enumerate(names):
            scorer = gradient_scorer if name in DIFFERENTIABLE_ALGORITHMS else black_box_scorer
            if progress is not None:
                if offset == 0:
                    progress.report()
                scorer.report_progress = progress.report
            started, cpu = time.perf_counter(), time.process_time()
            result = optimize(
                scorer, length, initial_values, objective, num_candidates,
                algorithm=name, alphabet=alphabet, constraint=constraint,
                seed=seed + offset, **shared_options,
            )
            completed.append((result, dict(
                seconds=time.perf_counter() - started,
                cpu_seconds=time.process_time() - cpu,
            )))
            if progress is not None:
                progress.completed += 1
                progress.report()

    for offset, (name, (raw_result, runtime)) in enumerate(zip(names, completed)):
        results = raw_result if isinstance(raw_result, list) else [raw_result]
        queries = sum(
            r.total_query_count if isinstance(r, SaturationResult)
            else r.metadata.get("total_convergence_queries", r.query_count)
            for r in results
        )
        reasons = {
            r.stop_reason if isinstance(r, SaturationResult)
            else r.metadata.get("stop_reason", "algorithm_stopped")
            for r in results
        }
        reason = ";".join(sorted(reasons)) or "algorithm_stopped"
        timings.append(dict(algorithm=name, **runtime, queries=queries, stop_reason=reason, seed=seed+offset))
        selection_details = []
        archives = []
        for reference_index, result in enumerate(results, start=1):
            if multiple_stages:
                archives.extend((reference_index, result.at(fraction))
                                for fraction in resolved.fractions)
            else:
                selected = result.selected if isinstance(result, SaturationResult) else result
                archives.append((reference_index, selected))
        for reference_index, result in archives:
            if not isinstance(result, OptimizationResult):
                raise TypeError(
                    "generate_designs requires optimizer results"
                )

            fields = ("selection", "saturation_fraction", "baseline_top_mean",
                      "plateau_top_mean", "selected_top_mean", "target_top_mean",
                      "score_threshold", "threshold_reached", "candidates_at_threshold",
                      "total_convergence_queries", "initial_cds")
            details = {key: result.metadata[key] for key in fields if key in result.metadata}
            details["selection_queries"] = result.query_count
            if multiple_stages:
                stage_timings.append(dict(
                    algorithm=name, reference_index=reference_index,
                    queries=details["total_convergence_queries"],
                    stop_reason=result.metadata["stop_reason"], **details,
                ))
            else:
                selection_details.append(details)
            result_task_names = result.task_names
            # Extract data from result
            for i, sequence in enumerate(result.sequences):
                row = {
                    "sequence": sequence,
                    "algorithm": name,
                    "score": float(result.scores[i]),
                }

                if multiple_stages:
                    row["gain_fraction"] = details["saturation_fraction"]
                # Add activity columns if available
                if result.activities is not None and result.task_names:
                    for j, task_name in enumerate(result.task_names):
                        row[f"activity_{task_name}"] = float(result.activities[i, j])

                all_rows.append(row)

        if len(selection_details) == 1:
            timings[-1].update(selection_details[0])
        elif selection_details:
            timings[-1]["reference_selections"] = selection_details

    frame = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=["sequence", "algorithm", "score"])
    frame["record_id"] = [f"D{i+1:05}" for i in range(len(frame))]
    frame["sequence_id"] = frame.sequence.map(lambda v: "S_"+hashlib.sha256(v.encode()).hexdigest()[:20])
    frame.attrs.update(objective=objective, task_names=list(result_task_names),
                       timing=timings, wall_seconds=time.perf_counter()-suite_started,
                       length=length, seed=seed, algorithms=list(names),
                       search_mode="threshold" if score_threshold is not None else "gain",
                       score_threshold=score_threshold,
                       gain_fraction=None if score_threshold is not None or multiple_stages else resolved.fractions[0],
                       candidates_per_algorithm=num_candidates, initial_count=len(initial_values),
                       minimum_hamming=diversity, execution="serial")
    if is_CDS_optimized:
        frame.attrs["cds_fixed_positions"] = resolve_fixed_positions(fixed_region, length)
    if multiple_stages:
        frame.attrs.update(gain_fractions=list(resolved.fractions), gain_stage_timing=stage_timings)
    if output_dir is not None:
        from .workflows import deduplicate_candidates, write_fasta
        output_dir.mkdir(parents=True, exist_ok=True)
        unique, _ = deduplicate_candidates(frame)
        frame.to_csv(output_dir/"designs.csv", index=False)
        unique.to_csv(output_dir/"unique_designs.csv", index=False)
        write_fasta(output_dir/"designs.fa", frame.record_id, frame.sequence)
        write_fasta(output_dir/"unique_designs.fa", unique.sequence_id, unique.sequence)
        pd.DataFrame(timings).to_csv(output_dir/"timing.csv", index=False)
        if multiple_stages:
            pd.DataFrame(stage_timings).to_csv(output_dir/"gain_stage_timing.csv", index=False)
        metadata = dict(frame.attrs)
        if isinstance(objective, CustomObjective):
            metadata["objective"] = objective.describe(len(result_task_names), result_task_names)
        elif is_dataclass(objective):
            metadata["objective"] = dict(type=type(objective).__name__, parameters=asdict(objective))
        (output_dir/"config.json").write_text(json.dumps(metadata, indent=2, default=lambda v: v.tolist())+"\n")
    return frame
