"""TFBind8 optimizer benchmark with algorithm-specific q99 saturation budgets.

TFBind8 has a finite 4^8 landscape.  Each fixed CNN weak ensemble is evaluated
over that landscape once and optimization sees only the resulting surrogate
table.  A seeded optimizer is run until the mean weak score of its best 100
novel designs stops materially improving.  Its final set is the first recorded
archive at 99% of that observed improvement.  Random screening is then run at
the median and arithmetic-mean q99 budgets of the eight seeded optimizers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from BioSeqBench.evaluation.evaluate_tfbind_candidates import evaluate_candidates
from BioSeqBench.models.tfbind8 import TFBindWeakEnsemble
from BioSeqBench.optimization import tfbind8_benchmark as base
from BioSeqBench.oracles.tfbind import codes_to_encoded, load_tfbind_landscape


FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SEEDED_ALGORITHMS = (
    "ism_greedy", "simulated_annealing", "genetic_algorithm", "adalead",
    "vanilla_activation_maximization", "fast_seqprop", "cbas", "den",
)
ALL_ALGORITHMS = (
    "training_top100", "random_screening_median_q99", "random_screening_mean_q99",
    *SEEDED_ALGORITHMS,
)


@dataclass(frozen=True)
class TFBind8SaturationConfig:
    train_fractions: tuple[float, ...] = FRACTIONS
    num_candidates: int = 100
    num_replicates: int = 5
    base_seed: int = 20260828
    top_visible_initializations: int = 50
    random_initializations: int = 50
    maximum_queries: int = 200_000
    minimum_queries: int = 10_000
    plateau_patience_queries: int = 10_000
    snapshot_interval_queries: int = 1_000
    material_improvement: float = 1e-5
    saturation_fraction: float = 0.99
    gradient_learning_rate: float = 0.1
    cbas_quantile: float = 0.8
    cbas_rounds: int = 20
    cbas_pretrain_epochs: int = 20
    cbas_update_epochs: int = 10
    den_diversity_weight: float = 0.1
    den_seed_pretrain_epochs: int = 200
    prediction_batch_size: int = 4096

    def validate(self) -> None:
        if not self.train_fractions or any(value <= 0 or value > .95 for value in self.train_fractions):
            raise ValueError("train_fractions must be non-empty and lie in (0, 0.95]")
        if self.num_candidates != 100:
            raise ValueError("This benchmark fixes the final candidate count at 100")
        if self.top_visible_initializations + self.random_initializations != self.num_candidates:
            raise ValueError("shared seeds must contain exactly 50 Train-TOP and 50 full-space random designs")
        if self.maximum_queries < self.minimum_queries or self.maximum_queries % self.num_candidates:
            raise ValueError("maximum_queries must be >= minimum_queries and divisible by 100")
        if self.snapshot_interval_queries <= 0 or self.plateau_patience_queries <= 0:
            raise ValueError("snapshot interval and plateau patience must be positive")
        if not 0 < self.saturation_fraction < 1:
            raise ValueError("saturation_fraction must lie in (0, 1)")


class SaturationReached(RuntimeError):
    """Private control-flow signal used to stop an optimizer after a plateau."""


@dataclass
class SaturatedSelection:
    codes: np.ndarray
    scores: np.ndarray
    stds: np.ndarray
    q99_query_count: int
    knee_query_count: int
    total_convergence_queries: int
    initial_top100_mean: float
    saturation_top100_mean: float
    q99_top100_mean: float
    stop_reason: str
    trajectory: pd.DataFrame


class SaturationTracker:
    """Online monotone top-100 archive tracker with a plateau stopping rule."""

    def __init__(self, visible_codes: set[int], config: TFBind8SaturationConfig) -> None:
        self.visible_codes = visible_codes
        self.config = config
        self.snapshots: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        self._next_snapshot = config.snapshot_interval_queries
        self._last_material_query: int | None = None
        self._last_material_value = -np.inf

    def observe(self, archive: base.CandidateArchive, query_count: int, *, force: bool = False) -> None:
        if not force and query_count < self._next_snapshot:
            return
        if query_count <= 0:
            return
        try:
            codes, scores, stds = archive.ranked(self.visible_codes, self.config.num_candidates)
        except RuntimeError:
            return
        if self.snapshots and query_count == self.snapshots[-1][0]:
            return
        self.snapshots.append((int(query_count), codes.copy(), scores.copy(), stds.copy()))
        self._next_snapshot = query_count + self.config.snapshot_interval_queries
        value = float(scores.mean())
        if value > self._last_material_value + self.config.material_improvement:
            self._last_material_value = value
            self._last_material_query = int(query_count)
        if (
            query_count >= self.config.minimum_queries
            and self._last_material_query is not None
            and query_count - self._last_material_query >= self.config.plateau_patience_queries
        ):
            raise SaturationReached

    def finalize(self, stop_reason: str) -> SaturatedSelection:
        if not self.snapshots:
            raise RuntimeError("Optimizer never produced 100 novel sequences outside the visible pool")
        x = np.asarray([item[0] for item in self.snapshots], dtype=np.int64)
        y = np.asarray([item[2].mean() for item in self.snapshots], dtype=np.float64)
        baseline, plateau = float(y[0]), float(y[-1])
        target = baseline + self.config.saturation_fraction * (plateau - baseline)
        index = int(np.flatnonzero(y >= target - 1e-12)[0])
        if len(y) == 1 or plateau - baseline <= self.config.material_improvement:
            knee_index = 0
        else:
            x_norm = (x - x[0]) / max(float(x[-1] - x[0]), 1.0)
            y_norm = (y - baseline) / max(plateau - baseline, 1e-12)
            knee_index = int(np.argmax(y_norm - x_norm))
        trajectory = pd.DataFrame({
            "query_count": x,
            "top100_mean_weak_prediction": y,
            "is_q99": np.arange(len(x)) == index,
            "is_knee": np.arange(len(x)) == knee_index,
            "saturation_target": target,
        })
        _, codes, scores, stds = self.snapshots[index]
        return SaturatedSelection(
            codes=codes, scores=scores, stds=stds,
            q99_query_count=int(x[index]), knee_query_count=int(x[knee_index]),
            total_convergence_queries=int(x[-1]), initial_top100_mean=baseline,
            saturation_top100_mean=plateau, q99_top100_mean=float(y[index]),
            stop_reason=stop_reason, trajectory=trajectory,
        )


@dataclass
class _Instrumentation:
    tracker: SaturationTracker
    config: TFBind8SaturationConfig
    current_query_count: int = 0


@contextmanager
def _instrument_archives(context: _Instrumentation) -> Iterator[None]:
    """Add archival tracing without changing the validated optimizer implementations."""
    original_archive = base.CandidateArchive
    original_surrogate = base.BudgetedLookupSurrogate

    class TrackingArchive(original_archive):
        def add(self, codes, scores, stds, step):  # type: ignore[no-untyped-def]
            super().add(codes, scores, stds, step)
            query_count = context.current_query_count or (int(step) + 1) * context.config.num_candidates
            context.tracker.observe(self, query_count)

    class TrackingSurrogate(original_surrogate):
        def query(self, codes):  # type: ignore[no-untyped-def]
            result = super().query(codes)
            context.current_query_count = self.query_count
            return result

    base.CandidateArchive = TrackingArchive  # type: ignore[assignment]
    base.BudgetedLookupSurrogate = TrackingSurrogate  # type: ignore[assignment]
    try:
        yield
    finally:
        base.CandidateArchive = original_archive  # type: ignore[assignment]
        base.BudgetedLookupSurrogate = original_surrogate  # type: ignore[assignment]


def _base_config(config: TFBind8SaturationConfig) -> base.TFBind8OptimizationConfig:
    return base.TFBind8OptimizationConfig(
        train_fractions=config.train_fractions, algorithms=SEEDED_ALGORITHMS,
        num_candidates=config.num_candidates, query_budget=config.maximum_queries,
        base_seed=config.base_seed, top_visible_initializations=config.top_visible_initializations,
        random_initializations=config.random_initializations,
        gradient_learning_rate=config.gradient_learning_rate,
        den_diversity_weight=config.den_diversity_weight, cbas_quantile=config.cbas_quantile,
        cbas_rounds=config.cbas_rounds, cbas_pretrain_epochs=config.cbas_pretrain_epochs,
        cbas_update_epochs=config.cbas_update_epochs, prediction_batch_size=config.prediction_batch_size,
        num_replicates=config.num_replicates, den_seed_pretrain_epochs=config.den_seed_pretrain_epochs,
    )


def _run_vanilla_with_tracking(
    ensemble: base.DifferentiableTFBindEnsemble, scores: np.ndarray, stds: np.ndarray,
    ranked_codes: np.ndarray, visible_count: int, length: int,
    config: base.TFBind8OptimizationConfig, seed: int, initial_codes: np.ndarray,
    tracker: SaturationTracker,
) -> None:
    """Equivalent to Vanilla AM, while recording exact discrete archive scores."""
    rng = base._set_seed(seed)
    device = ensemble.label_means.device
    visible = set(map(int, ranked_codes[:visible_count]))
    initial = base._resolve_initial_codes(initial_codes, ranked_codes, visible_count, len(scores), config, rng)
    logits = torch.nn.Parameter(base._one_hot_for_codes(initial, length, device) * 1.5 + .05 * torch.randn((len(initial), length, 4), device=device))
    optimizer = torch.optim.Adam([logits], lr=config.gradient_learning_rate)
    archive = base.CandidateArchive()
    proxy_records: dict[int, float] = {}
    gradient_steps = (config.query_budget - config.num_candidates) // config.num_candidates
    executed_queries = 0
    try:
        for step in range(gradient_steps):
            optimizer.zero_grad(set_to_none=True)
            probabilities = torch.softmax(logits, dim=-1)
            prediction, _ = ensemble(probabilities)
            (-prediction.mean()).backward()
            optimizer.step()
            codes = base._codes_for_one_hot(probabilities)
            exact = scores[codes]
            executed_queries = (step + 1) * config.num_candidates
            archive.add(codes, exact, stds[codes], step=step)
            for code, value in zip(codes, prediction.detach().cpu().numpy()):
                proxy_records[int(code)] = max(float(value), proxy_records.get(int(code), -np.inf))
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=-1)
            final_proxy = ensemble(probabilities)[0].detach().cpu().numpy()
            attempts = 0
            while len([code for code in proxy_records if code not in visible]) < config.num_candidates and attempts < 1000:
                sampled = torch.multinomial(probabilities.reshape(-1, 4), 1).reshape(config.num_candidates, length)
                sampled_codes = base.encoded_to_codes(sampled.cpu().numpy().astype(np.uint8))
                for code, value in zip(sampled_codes, final_proxy):
                    proxy_records[int(code)] = max(float(value), proxy_records.get(int(code), -np.inf))
                attempts += 1
        if len([code for code in proxy_records if code not in visible]) < config.num_candidates:
            available = np.asarray([code for code in range(len(scores)) if code not in visible and code not in proxy_records])
            for code in rng.choice(available, size=config.num_candidates - len([code for code in proxy_records if code not in visible]), replace=False):
                proxy_records[int(code)] = -np.inf
        selected = base._rank_proxy_pool(proxy_records, visible, config.num_candidates)
        executed_queries = config.query_budget
        archive.add(selected, scores[selected], stds[selected], step=gradient_steps)
    finally:
        tracker.observe(archive, executed_queries, force=True)


def _run_seeded_optimizer(
    algorithm: str, *, ensemble: base.DifferentiableTFBindEnsemble, scores: np.ndarray,
    stds: np.ndarray, ranked_codes: np.ndarray, visible_count: int, length: int,
    config: TFBind8SaturationConfig, seed: int, initial_codes: np.ndarray, device: torch.device,
) -> SaturatedSelection:
    optimizer_config = _base_config(config)
    tracker = SaturationTracker(set(map(int, ranked_codes[:visible_count])), config)
    instrumentation = _Instrumentation(tracker, config)
    stop_reason = "maximum_queries"
    try:
        with _instrument_archives(instrumentation):
            if algorithm in base.BLACK_BOX_OPTIMIZERS:
                base.BLACK_BOX_OPTIMIZERS[algorithm](scores, stds, ranked_codes, visible_count, length, optimizer_config, seed, initial_codes)
            elif algorithm == "vanilla_activation_maximization":
                _run_vanilla_with_tracking(ensemble, scores, stds, ranked_codes, visible_count, length, optimizer_config, seed, initial_codes, tracker)
            elif algorithm == "fast_seqprop":
                base.optimize_fast_seqprop(ensemble, ranked_codes, visible_count, len(scores), length, optimizer_config, seed, initial_codes)
            elif algorithm == "cbas":
                base.optimize_cbas(scores, stds, ranked_codes, visible_count, length, optimizer_config, seed, device, initial_codes)
            elif algorithm == "den":
                base.optimize_den(ensemble, ranked_codes, visible_count, length, optimizer_config, seed, initial_codes)
            else:  # pragma: no cover - caller iterates a fixed tuple.
                raise ValueError(algorithm)
    except SaturationReached:
        stop_reason = "post_knee_plateau"
    return tracker.finalize(stop_reason)


def _score_selection(
    algorithm: str, selection: SaturatedSelection, landscape, fraction: float, visible_codes: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sequences = landscape.sequences_for_codes(selection.codes).tolist()
    candidates, exact = evaluate_candidates("tfbind8", sequences, train_fraction=fraction, top_k=100)
    candidates["sequence_code"] = selection.codes
    candidates["weak_prediction"] = selection.scores
    candidates["weak_uncertainty"] = selection.stds
    candidates["prediction_error"] = selection.scores - candidates["true_score"].to_numpy()
    candidates["algorithm"] = algorithm
    candidates["weak_model_fraction"] = fraction
    candidates["query_count"] = selection.q99_query_count
    encoded = codes_to_encoded(selection.codes, landscape.length)
    train_encoded = codes_to_encoded(visible_codes, landscape.length)
    minimum_distance = np.full(len(encoded), landscape.length, dtype=np.int64)
    for start in range(0, len(train_encoded), 4096):
        minimum_distance = np.minimum(minimum_distance, (encoded[:, None, :] != train_encoded[None, start:start + 4096, :]).sum(-1).min(1))
    candidates["minimum_training_hamming"] = minimum_distance
    truth = candidates["true_score"].to_numpy()
    percentiles = candidates["activity_percentile"].to_numpy()
    summary: dict[str, object] = {
        **exact, "algorithm": algorithm, "weak_model_fraction": fraction,
        "query_count": selection.q99_query_count, "mean_weak_prediction": float(selection.scores.mean()),
        "max_weak_prediction": float(selection.scores.max()), "mean_prediction_error": float((selection.scores - truth).mean()),
        "mean_absolute_prediction_error": float(np.abs(selection.scores - truth).mean()),
        "median_true_score": float(np.median(truth)), "top_10_mean_true_score": float(np.sort(truth)[-10:].mean()),
        "mean_activity_percentile": float(percentiles.mean()), "top_5pct_count": int((percentiles >= 95).sum()),
        "top_1pct_count": int((percentiles >= 99).sum()), "top_0_1pct_count": int((percentiles >= 99.9).sum()),
        "mean_pairwise_hamming": float((encoded[:, None, :] != encoded[None, :, :]).sum(-1)[np.triu_indices(len(encoded), 1)].mean()),
        "mean_minimum_training_hamming": float(minimum_distance.mean()),
        "q99_query_count": selection.q99_query_count, "knee_query_count": selection.knee_query_count,
        "total_convergence_queries": selection.total_convergence_queries,
        "initial_top100_mean": selection.initial_top100_mean, "saturation_top100_mean": selection.saturation_top100_mean,
        "q99_top100_mean": selection.q99_top100_mean, "stop_reason": selection.stop_reason,
    }
    return candidates, summary


def _random_selection(scores: np.ndarray, stds: np.ndarray, visible: set[int], budget: int, seed: int) -> SaturatedSelection:
    rng = np.random.default_rng(seed)
    available = np.asarray([code for code in range(len(scores)) if code not in visible], dtype=np.int64)
    sampled = rng.choice(available, size=budget, replace=budget > len(available))
    best = sorted(set(map(int, sampled)), key=lambda code: (-scores[code], code))[:100]
    if len(best) < 100:
        raise RuntimeError("Random screening did not produce 100 unique novel candidates")
    selected = np.asarray(best, dtype=np.int64)
    value = float(scores[selected].mean())
    trajectory = pd.DataFrame({"query_count": [budget], "top100_mean_weak_prediction": [value], "is_q99": [True], "is_knee": [False], "saturation_target": [value]})
    return SaturatedSelection(selected, scores[selected], stds[selected], budget, 0, budget, value, value, value, "matched_seeded_optimizer_q99_budget", trajectory)


def _training_reference(scores: np.ndarray, stds: np.ndarray, ranked_codes: np.ndarray, visible_count: int) -> SaturatedSelection:
    codes = ranked_codes[visible_count - 100:visible_count][::-1].astype(np.int64)
    value = float(scores[codes].mean())
    return SaturatedSelection(codes, scores[codes], stds[codes], 0, 0, 0, value, value, value, "observed_reference", pd.DataFrame())


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ("mean_true_score", "median_true_score", "top_10_mean_true_score", "best_true_score", "top_5pct_count", "top_1pct_count", "top_0_1pct_count", "mean_pairwise_hamming", "mean_absolute_prediction_error", "mean_activity_percentile")
    named = {f"{metric}_{stat}": (metric, stat) for metric in metrics for stat in ("mean", "std", "sem")}
    return summary.groupby(["dataset", "weak_model_fraction", "algorithm"], sort=False).agg(replicates=("replicate", "size"), **named).reset_index()


def run_tfbind8_saturation_benchmark(
    *, checkpoint_root: str | Path = "BioSeqBench/results/tfbind_weak_cnns/tfbind8",
    prediction_cache_root: str | Path = "BioSeqBench/results/tfbind8_optimization_all_models_5seeds_shared_init",
    output_dir: str | Path = "BioSeqBench/results/tfbind8_optimization_saturation_5seeds",
    raw_path: str | Path | None = None, device: str | torch.device | None = None,
    config: TFBind8SaturationConfig | None = None, resume: bool = False,
) -> pd.DataFrame:
    config = config or TFBind8SaturationConfig()
    config.validate()
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    landscape = load_tfbind_landscape("tfbind8", raw_path)
    ranked_codes = landscape.ranked_codes()
    protocol = {**asdict(config), "train_fractions": list(config.train_fractions), "seeded_algorithms": list(SEEDED_ALGORITHMS), "all_algorithms": list(ALL_ALGORITHMS),
                "selection_rule": "earliest top-100 novel archive reaching baseline + 0.99*(observed plateau-baseline)",
                "knee_rule": "maximum normalized vertical distance y_norm-x_norm on monotone top-100 curve",
                "plateau_rule": "no cumulative top-100 mean improvement >= material_improvement for plateau_patience_queries",
                "random_rule": "per model/replicate median and arithmetic mean of eight seeded-algorithm q99 counts",
                "evaluation_policy": "optimization uses fixed CNN weak ensemble; all final sequences receive exact TFBind8 experimental scores",
                "initialization_policy": "all eight seeded optimizers share 50 highest-visible training sequences + 50 full-space uniform random sequences for each condition; Random has no initialization"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries: list[dict[str, object]] = []; candidates_all: list[pd.DataFrame] = []; initializations: list[pd.DataFrame] = []
    for fraction_index, fraction in enumerate(config.train_fractions):
        name = base._fraction_name(fraction); model_dir = output / name; model_dir.mkdir(exist_ok=True)
        cache = Path(prediction_cache_root) / name / "all_landscape_weak_predictions.npz"
        ensemble = base.TFBindWeakEnsemble(Path(checkpoint_root) / name, device=device)
        scores, stds = base._load_or_compute_predictions(ensemble, landscape, cache, config.prediction_batch_size)
        differentiable = base.DifferentiableTFBindEnsemble(ensemble).to(device)
        visible_count = int(round(landscape.size * fraction)); visible_codes = ranked_codes[:visible_count]; visible_set = set(map(int, visible_codes))
        print(f"Loaded {name} ({visible_count:,} visible designs) on {device}", flush=True)
        for replicate in range(config.num_replicates):
            run_dir = model_dir / f"run_{replicate:02d}"; run_dir.mkdir(exist_ok=True)
            init_seed = config.base_seed + fraction_index * 100_000 + replicate * 1_000 + 997
            initial_codes, sources = base._shared_initial_codes(ranked_codes, visible_count, landscape.size, _base_config(config), np.random.default_rng(init_seed))
            init_id = hashlib.sha256(initial_codes.tobytes()).hexdigest()[:16]
            initial = pd.DataFrame({"sequence_code": initial_codes, "sequence": landscape.sequences_for_codes(initial_codes), "source": sources, "in_weak_train": np.isin(initial_codes, visible_codes), "weak_prediction": scores[initial_codes], "true_score": landscape.labels_by_code[initial_codes], "weak_model_fraction": fraction, "replicate": replicate, "initialization_seed": init_seed, "initialization_id": init_id})
            initial.to_csv(run_dir / "shared_initial_sequences.csv", index=False); initializations.append(initial)
            seeded: dict[str, SaturatedSelection] = {}
            for algorithm_index, algorithm in enumerate(SEEDED_ALGORITHMS):
                candidate_path = run_dir / f"{algorithm}_candidates.csv"; summary_path = run_dir / f"{algorithm}_summary.json"; trajectory_path = run_dir / f"{algorithm}_trajectory.csv"
                previous = json.loads(summary_path.read_text()) if summary_path.exists() else None
                can_resume = (
                    resume and candidate_path.exists() and trajectory_path.exists() and previous is not None
                    and not (
                        previous.get("stop_reason") == "maximum_queries"
                        and int(previous.get("total_convergence_queries", 0)) < config.maximum_queries
                    )
                )
                if can_resume:
                    scored = pd.read_csv(candidate_path); summary = previous; trajectory = pd.read_csv(trajectory_path)
                    seeded[algorithm] = SaturatedSelection(scored.sequence_code.to_numpy(), scored.weak_prediction.to_numpy(), scored.weak_uncertainty.to_numpy(), int(summary["q99_query_count"]), int(summary["knee_query_count"]), int(summary["total_convergence_queries"]), float(summary["initial_top100_mean"]), float(summary["saturation_top100_mean"]), float(summary["q99_top100_mean"]), str(summary["stop_reason"]), trajectory)
                else:
                    seed = config.base_seed + fraction_index * 100_000 + replicate * 1_000 + algorithm_index * 101
                    print(f"Saturating {name} run={replicate:02d} {algorithm}", flush=True)
                    seeded[algorithm] = _run_seeded_optimizer(algorithm, ensemble=differentiable, scores=scores, stds=stds, ranked_codes=ranked_codes, visible_count=visible_count, length=landscape.length, config=config, seed=seed, initial_codes=initial_codes, device=device)
                    seeded[algorithm].trajectory.to_csv(trajectory_path, index=False)
                    scored, summary = _score_selection(algorithm, seeded[algorithm], landscape, fraction, visible_codes)
                    scored.to_csv(candidate_path, index=False); summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else summary
                scored = pd.read_csv(candidate_path)
                summary.update({"dataset": "tfbind8", "replicate": replicate, "optimizer_seed": config.base_seed + fraction_index * 100_000 + replicate * 1_000 + algorithm_index * 101, "initialization_seed": init_seed, "initialization_id": init_id, "uses_shared_initialization": True})
                summaries.append(summary); candidates_all.append(scored.assign(replicate=replicate, initialization_id=init_id))
            q99 = np.asarray([selection.q99_query_count for selection in seeded.values()], dtype=np.int64)
            for random_index, (algorithm, budget) in enumerate((("random_screening_median_q99", int(round(np.median(q99)))), ("random_screening_mean_q99", int(round(q99.mean()))))):
                path = run_dir / f"{algorithm}_candidates.csv"; summary_path = run_dir / f"{algorithm}_summary.json"
                if resume and path.exists() and summary_path.exists() and int(json.loads(summary_path.read_text())["q99_query_count"]) == budget:
                    scored = pd.read_csv(path); summary = json.loads(summary_path.read_text())
                else:
                    selection = _random_selection(scores, stds, visible_set, budget, config.base_seed + fraction_index * 100_000 + replicate * 1_000 + 900 + random_index)
                    scored, summary = _score_selection(algorithm, selection, landscape, fraction, visible_codes)
                    scored.to_csv(path, index=False); summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                summary.update({"dataset": "tfbind8", "replicate": replicate, "optimizer_seed": config.base_seed + fraction_index * 100_000 + replicate * 1_000 + 900 + random_index, "initialization_seed": np.nan, "initialization_id": "none", "uses_shared_initialization": False})
                summaries.append(summary); candidates_all.append(scored.assign(replicate=replicate, initialization_id="none"))
            reference = _training_reference(scores, stds, ranked_codes, visible_count)
            path = run_dir / "training_top100_candidates.csv"; summary_path = run_dir / "training_top100_summary.json"
            scored, summary = _score_selection("training_top100", reference, landscape, fraction, visible_codes)
            scored.to_csv(path, index=False); summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            summary.update({"dataset": "tfbind8", "replicate": replicate, "optimizer_seed": np.nan, "initialization_seed": np.nan, "initialization_id": "none", "uses_shared_initialization": False})
            summaries.append(summary); candidates_all.append(scored.assign(replicate=replicate, initialization_id="none"))
    summary_frame = pd.DataFrame(summaries); candidate_frame = pd.concat(candidates_all, ignore_index=True); init_frame = pd.concat(initializations, ignore_index=True)
    summary_frame.to_csv(output / "summary.csv", index=False); candidate_frame.to_csv(output / "all_candidates.csv", index=False); init_frame.to_csv(output / "all_shared_initial_sequences.csv", index=False)
    convergence = summary_frame[summary_frame.algorithm.isin(SEEDED_ALGORITHMS)][["weak_model_fraction", "replicate", "algorithm", "q99_query_count", "knee_query_count", "total_convergence_queries", "initial_top100_mean", "saturation_top100_mean", "q99_top100_mean", "stop_reason"]]
    convergence.to_csv(output / "convergence_summary.csv", index=False); _aggregate(summary_frame).to_csv(output / "aggregate_summary.csv", index=False)
    return summary_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="BioSeqBench/results/tfbind8_optimization_saturation_5seeds")
    parser.add_argument("--checkpoint-root", default="BioSeqBench/results/tfbind_weak_cnns/tfbind8")
    parser.add_argument("--prediction-cache-root", default="BioSeqBench/results/tfbind8_optimization_all_models_5seeds_shared_init")
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fractions", nargs="*", type=float, default=None); parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--maximum-queries", type=int, default=200_000)
    args = parser.parse_args()
    config = TFBind8SaturationConfig(train_fractions=tuple(args.fractions) if args.fractions else FRACTIONS, num_replicates=args.replicates, maximum_queries=args.maximum_queries)
    summary = run_tfbind8_saturation_benchmark(checkpoint_root=args.checkpoint_root, prediction_cache_root=args.prediction_cache_root, output_dir=args.output_dir, device=args.device, config=config, resume=args.resume)
    print(summary[["weak_model_fraction", "replicate", "algorithm", "q99_query_count", "mean_true_score", "best_true_score", "stop_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
