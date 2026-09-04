"""Optimus 5'UTR benchmark with optimizer-specific saturation budgets.

Each seeded optimizer runs until its monotone top-100 weak-score curve reaches a
stable post-knee plateau.  Candidates are selected retrospectively at the first
query count reaching 90% of the observed gain from the first valid top-100
archive to that plateau.  Random screening uses the per-condition median and
mean q90 counts of the eight seeded optimizers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from torch import nn
from torch.nn import functional as F

from BioSeqBench.models.utr5_mpra import DifferentiableUTR5Ensemble, UTR5WeakEnsemble
from BioSeqBench.optimization.tfbind8_benchmark import (
    AutoregressiveDNA,
    DeepExplorationGenerator,
    _adalead_parent_mask,
    _fast_seqprop_logits,
    _fit_autoregressive,
    _straight_through_sample,
)
from BioSeqBench.optimization.utr5_benchmark import (
    CandidateArchive,
    OptimizationResult,
    _aggregate,
    _encoded_to_sequences,
    _evaluate_result,
    _load_reference_scores,
    _one_hot_for_sequences,
    _random_sequences,
    _random_single_mutants,
    _sequences_for_one_hot,
    _sequences_to_encoded,
    _set_seed,
    _shared_initial_sequences,
)
from BioSeqBench.oracles import Optimus5PrimeOracle
from BioSeqBench.training.train_utr5_weak_cnns import _index_digest, _split_visible_pool


FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SEEDED_ALGORITHMS = (
    "ism_greedy",
    "simulated_annealing",
    "genetic_algorithm",
    "adalead",
    "vanilla_activation_maximization",
    "fast_seqprop",
    "cbas",
    "den",
)
ALGORITHMS = (
    "training_top100",
    "random_screening_median_q90",
    "random_screening_mean_q90",
    *SEEDED_ALGORITHMS,
)


@dataclass(frozen=True)
class SaturationConfig:
    train_fractions: tuple[float, ...] = FRACTIONS
    num_candidates: int = 100
    num_replicates: int = 5
    base_seed: int = 20260827
    top_visible_initializations: int = 50
    random_initializations: int = 50
    prediction_batch_size: int = 2048
    minimum_queries: int = 20_000
    plateau_patience_queries: int = 50_000
    maximum_queries: int = 100_000_000
    max_runtime_seconds: float = 900.0
    material_improvement: float = 1e-4
    saturation_fraction: float = 0.90
    gradient_learning_rate: float = 0.1
    cbas_quantile: float = 0.8
    cbas_batch_size: int = 1000
    cbas_pretrain_epochs: int = 20
    cbas_update_epochs: int = 10
    den_diversity_weight: float = 0.1
    den_seed_pretrain_epochs: int = 200

    def validate(self) -> None:
        if not self.train_fractions or any(value <= 0 or value > 0.95 for value in self.train_fractions):
            raise ValueError("train fractions must lie in (0, 0.95]")
        if self.num_candidates != 100:
            raise ValueError("saturation benchmark currently requires exactly 100 candidates")
        if self.top_visible_initializations + self.random_initializations != 100:
            raise ValueError("initialization must contain exactly 100 sequences")
        if self.num_replicates < 1 or self.prediction_batch_size < 1:
            raise ValueError("replicate count and prediction batch size must be positive")
        if not 0 < self.saturation_fraction < 1:
            raise ValueError("saturation_fraction must lie in (0, 1)")
        if not (0 < self.minimum_queries <= self.maximum_queries):
            raise ValueError("invalid minimum/maximum query limits")
        if self.plateau_patience_queries < 1 or self.material_improvement <= 0:
            raise ValueError("plateau patience and material improvement must be positive")
        if self.max_runtime_seconds <= 0:
            raise ValueError("max runtime must be positive")


class OnlineSurrogate:
    def __init__(
        self,
        ensemble: UTR5WeakEnsemble,
        config: SaturationConfig,
        *,
        started_at: float | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.config = config
        self.query_count = 0
        self.started_at = time.monotonic() if started_at is None else started_at

    @property
    def remaining(self) -> int:
        return self.config.maximum_queries - self.query_count

    @property
    def runtime_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def runtime_limit_reached(self) -> bool:
        return self.runtime_seconds >= self.config.max_runtime_seconds

    def query(self, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(sequences, dtype=str)
        if len(values) > self.remaining:
            raise RuntimeError("maximum weak-model query count exceeded")
        scores, stds = self.ensemble.predict(values, batch_size=self.config.prediction_batch_size)
        self.query_count += len(values)
        return scores.astype(np.float64), stds.astype(np.float64)


@dataclass
class SaturatedResult:
    result: OptimizationResult
    trajectory: pd.DataFrame
    q90_query_count: int
    knee_query_count: int
    total_convergence_queries: int
    initial_top100_mean: float
    saturation_top100_mean: float
    q90_top100_mean: float
    stop_reason: str


class SaturationTracker:
    """Track a monotone archive curve and retain top-100 improvement snapshots."""

    def __init__(self, visible_set: set[str], config: SaturationConfig) -> None:
        self.visible_set = visible_set
        self.config = config
        self.rows: list[dict[str, float | int]] = []
        self.snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.top_records: dict[str, tuple[float, float]] = {}
        self.material_level = -np.inf
        self.last_material_query = 0
        self.last_curve_value = -np.inf

    def observe_batch(
        self,
        query_count: int,
        sequences: np.ndarray,
        scores: np.ndarray,
        stds: np.ndarray,
    ) -> None:
        combined = dict(self.top_records)
        for sequence, score, std in zip(sequences, scores, stds):
            key = str(sequence)
            if key in self.visible_set:
                continue
            value = (float(score), float(std))
            if key not in combined or value[0] > combined[key][0]:
                combined[key] = value
        if len(combined) < self.config.num_candidates:
            self.top_records = combined
            return
        selected = sorted(combined.items(), key=lambda item: (-item[1][0], item[0]))[: self.config.num_candidates]
        self.top_records = dict(selected)
        sequences = np.asarray([item[0] for item in selected], dtype=str)
        scores = np.asarray([item[1][0] for item in selected], dtype=np.float64)
        stds = np.asarray([item[1][1] for item in selected], dtype=np.float64)
        top_mean = float(scores.mean())
        best = float(scores.max())
        if top_mean > self.material_level + self.config.material_improvement:
            self.material_level = top_mean
            self.last_material_query = query_count
        if top_mean > self.last_curve_value + 1e-12:
            self.last_curve_value = top_mean
            self.rows.append(
                {"query_count": query_count, "top100_mean_weak_prediction": top_mean, "best_weak_prediction": best}
            )
            self.snapshots.append((sequences.copy(), scores.copy(), stds.copy()))

    def observe(self, query_count: int, archive: CandidateArchive) -> None:
        """Compatibility helper for tests and external archive callers."""
        sequences = np.asarray(list(archive.records), dtype=str)
        values = list(archive.records.values())
        scores = np.asarray([value[0] for value in values], dtype=np.float64)
        stds = np.asarray([value[1] for value in values], dtype=np.float64)
        self.observe_batch(query_count, sequences, scores, stds)

    def ranked(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.top_records) < self.config.num_candidates:
            raise RuntimeError("fewer than 100 novel candidates have been tracked")
        selected = sorted(self.top_records.items(), key=lambda item: (-item[1][0], item[0]))
        return (
            np.asarray([item[0] for item in selected], dtype=str),
            np.asarray([item[1][0] for item in selected], dtype=np.float64),
            np.asarray([item[1][1] for item in selected], dtype=np.float64),
        )

    def plateau_reached(self, query_count: int) -> bool:
        return (
            bool(self.rows)
            and query_count >= self.config.minimum_queries
            and query_count - self.last_material_query >= self.config.plateau_patience_queries
        )

    def finalize(self, algorithm: str, total_queries: int, stop_reason: str, metadata: dict) -> SaturatedResult:
        if not self.rows:
            raise RuntimeError(f"{algorithm} never accumulated 100 novel candidates")
        trajectory = pd.DataFrame(self.rows)
        x = trajectory["query_count"].to_numpy(dtype=float)
        y = trajectory["top100_mean_weak_prediction"].to_numpy(dtype=float)
        baseline, saturation = float(y[0]), float(y[-1])
        target = baseline + self.config.saturation_fraction * (saturation - baseline)
        q90_index = int(np.flatnonzero(y >= target - 1e-12)[0])
        if len(y) == 1 or saturation - baseline <= self.config.material_improvement:
            knee_index = 0
        else:
            x_norm = (x - x[0]) / max(x[-1] - x[0], 1.0)
            y_norm = (y - baseline) / max(saturation - baseline, 1e-12)
            knee_index = int(np.argmax(y_norm - x_norm))
        sequences, scores, stds = self.snapshots[q90_index]
        result_metadata = {
            **metadata,
            "selection_rule": (
                f"earliest top-100 archive reaching {self.config.saturation_fraction:.0%} "
                "of observed saturation gain"
            ),
            "q90_query_count": int(x[q90_index]),
            "knee_query_count": int(x[knee_index]),
            "total_convergence_queries": int(total_queries),
            "initial_top100_mean": baseline,
            "saturation_top100_mean": saturation,
            "q90_top100_mean": float(y[q90_index]),
            "stop_reason": stop_reason,
        }
        result = OptimizationResult(
            algorithm, sequences, scores, stds, int(x[q90_index]), result_metadata
        )
        trajectory["is_knee"] = False
        trajectory.loc[knee_index, "is_knee"] = True
        trajectory["is_q90"] = False
        trajectory.loc[q90_index, "is_q90"] = True
        trajectory["saturation_target"] = target
        return SaturatedResult(
            result=result,
            trajectory=trajectory,
            q90_query_count=int(x[q90_index]),
            knee_query_count=int(x[knee_index]),
            total_convergence_queries=int(total_queries),
            initial_top100_mean=baseline,
            saturation_top100_mean=saturation,
            q90_top100_mean=float(y[q90_index]),
            stop_reason=stop_reason,
        )


def _add_and_observe(
    tracker: SaturationTracker,
    surrogate: OnlineSurrogate,
    sequences: np.ndarray,
    scores: np.ndarray,
    stds: np.ndarray,
    step: int,
) -> None:
    tracker.observe_batch(surrogate.query_count, sequences, scores, stds)


def adaptive_ism(
    ensemble: UTR5WeakEnsemble, initial: np.ndarray, visible_set: set[str], config: SaturationConfig, seed: int
) -> SaturatedResult:
    rng = _set_seed(seed)
    surrogate, tracker = OnlineSurrogate(ensemble, config), SaturationTracker(visible_set, config)
    current = initial.copy()
    current_scores, current_stds = surrogate.query(current)
    _add_and_observe(tracker, surrogate, current, current_scores, current_stds, 0)
    sweeps = neighborhoods = 0
    reason = "emergency_query_ceiling"
    while surrogate.remaining >= 150:
        progressed = False
        indices = rng.permutation(len(current))[: surrogate.remaining // 150]
        sweep_mutants: list[np.ndarray] = []
        for index in indices:
            encoded = _sequences_to_encoded([current[index]])[0]
            mutants = np.repeat(encoded[None, :], 150, axis=0)
            row = 0
            for position in range(50):
                for base in range(4):
                    if base != encoded[position]:
                        mutants[row, position] = base
                        row += 1
            sweep_mutants.append(mutants)
        mutant_sequences = _encoded_to_sequences(np.concatenate(sweep_mutants))
        scores, stds = surrogate.query(mutant_sequences)
        neighborhoods += len(indices)
        _add_and_observe(tracker, surrogate, mutant_sequences, scores, stds, neighborhoods)
        for offset, index in enumerate(indices):
            block = slice(offset * 150, (offset + 1) * 150)
            best_offset = int(np.argmax(scores[block]))
            best = offset * 150 + best_offset
            if scores[best] > current_scores[index] + 1e-12:
                current[index], current_scores[index], current_stds[index] = mutant_sequences[best], scores[best], stds[best]
                progressed = True
        sweeps += 1
        if len(indices) == len(current) and not progressed:
            reason = "all_100_seeds_single_mutation_local_optima"
            break
        if tracker.plateau_reached(surrogate.query_count):
            reason = "post_knee_plateau"
            break
        if surrogate.runtime_limit_reached:
            reason = "runtime_limit"
            break
    return tracker.finalize("ism_greedy", surrogate.query_count, reason, {
        "sweeps": sweeps, "neighborhoods": neighborhoods, "runtime_seconds": surrogate.runtime_seconds,
    })


def adaptive_sa(
    ensemble: UTR5WeakEnsemble, initial: np.ndarray, visible_set: set[str], config: SaturationConfig, seed: int
) -> SaturatedResult:
    rng = _set_seed(seed)
    surrogate, tracker = OnlineSurrogate(ensemble, config), SaturationTracker(visible_set, config)
    current = initial.copy()
    current_scores, current_stds = surrogate.query(current)
    _add_and_observe(tracker, surrogate, current, current_scores, current_stds, 0)
    temperature0 = max(float(np.std(current_scores)), 1e-3)
    step = 0
    reason = "emergency_query_ceiling"
    while surrogate.remaining >= len(current):
        proposals = _random_single_mutants(current, rng)
        scores, stds = surrogate.query(proposals)
        step += 1
        _add_and_observe(tracker, surrogate, proposals, scores, stds, step)
        progress = surrogate.query_count / config.maximum_queries
        temperature = max(temperature0 * (0.001 ** progress), 1e-6)
        delta = scores - current_scores
        accept = (delta >= 0) | (rng.random(len(current)) < np.exp(np.minimum(delta / temperature, 0)))
        current[accept], current_scores[accept], current_stds[accept] = proposals[accept], scores[accept], stds[accept]
        if tracker.plateau_reached(surrogate.query_count):
            reason = "post_knee_plateau"
            break
        if surrogate.runtime_limit_reached:
            reason = "runtime_limit"
            break
    return tracker.finalize("simulated_annealing", surrogate.query_count, reason, {
        "steps": step, "initial_temperature": temperature0, "runtime_seconds": surrogate.runtime_seconds,
    })


def adaptive_ga(
    ensemble: UTR5WeakEnsemble, initial: np.ndarray, visible_set: set[str], config: SaturationConfig, seed: int
) -> SaturatedResult:
    rng = _set_seed(seed)
    surrogate, tracker = OnlineSurrogate(ensemble, config), SaturationTracker(visible_set, config)
    population = initial.copy()
    scores, stds = surrogate.query(population)
    _add_and_observe(tracker, surrogate, population, scores, stds, 0)
    elite_count = 20
    generation = 0
    reason = "emergency_query_ceiling"
    while surrogate.remaining >= 100:
        elites = _sequences_to_encoded(population[np.argsort(scores)[-elite_count:]])
        parent_a = elites[rng.integers(0, elite_count, size=100)]
        parent_b = elites[rng.integers(0, elite_count, size=100)]
        children = np.where(rng.random(parent_a.shape) < 0.5, parent_a, parent_b)
        mutation = rng.random(children.shape) < 0.02
        children = np.where(mutation, (children + rng.integers(1, 4, children.shape)) % 4, children)
        child_sequences = _encoded_to_sequences(children)
        child_scores, child_stds = surrogate.query(child_sequences)
        generation += 1
        _add_and_observe(tracker, surrogate, child_sequences, child_scores, child_stds, generation)
        all_sequences = np.concatenate([population, child_sequences])
        all_scores = np.concatenate([scores, child_scores])
        all_stds = np.concatenate([stds, child_stds])
        keep = np.argsort(all_scores)[-100:]
        population, scores, stds = all_sequences[keep], all_scores[keep], all_stds[keep]
        if tracker.plateau_reached(surrogate.query_count):
            reason = "post_knee_plateau"
            break
        if surrogate.runtime_limit_reached:
            reason = "runtime_limit"
            break
    return tracker.finalize("genetic_algorithm", surrogate.query_count, reason, {
        "generations": generation, "elite_count": elite_count, "runtime_seconds": surrogate.runtime_seconds,
    })


def adaptive_adalead(
    ensemble: UTR5WeakEnsemble, initial: np.ndarray, visible_set: set[str], config: SaturationConfig, seed: int
) -> SaturatedResult:
    rng = _set_seed(seed)
    surrogate, tracker = OnlineSurrogate(ensemble, config), SaturationTracker(visible_set, config)
    population = initial.copy()
    scores, stds = surrogate.query(population)
    _add_and_observe(tracker, surrogate, population, scores, stds, 0)
    roots = population[_adalead_parent_mask(scores, 0.05)]
    root_scores = scores[_adalead_parent_mask(scores, 0.05)]
    nodes, node_roots = np.resize(roots, 100), np.resize(root_scores, 100)
    step = 0
    reason = "emergency_query_ceiling"
    while surrogate.remaining >= 100:
        children = _random_single_mutants(nodes, rng)
        child_scores, child_stds = surrogate.query(children)
        step += 1
        _add_and_observe(tracker, surrogate, children, child_scores, child_stds, step)
        continued = child_scores >= node_roots
        nodes[continued] = children[continued]
        reset_count = int((~continued).sum())
        if reset_count:
            nodes[~continued] = np.resize(roots, reset_count)
            node_roots[~continued] = np.resize(root_scores, reset_count)
        if step % 10 == 0:
            population, scores, stds = tracker.ranked()
            roots = population[_adalead_parent_mask(scores, 0.05)]
            root_scores = scores[_adalead_parent_mask(scores, 0.05)]
        if tracker.plateau_reached(surrogate.query_count):
            reason = "post_knee_plateau"
            break
        if surrogate.runtime_limit_reached:
            reason = "runtime_limit"
            break
    return tracker.finalize("adalead", surrogate.query_count, reason, {
        "steps": step, "threshold": 0.05, "runtime_seconds": surrogate.runtime_seconds,
    })


def adaptive_gradient(
    name: str,
    differentiable: DifferentiableUTR5Ensemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: SaturationConfig,
    seed: int,
) -> SaturatedResult:
    _set_seed(seed)
    started_at = time.monotonic()
    device = differentiable.label_means.device
    initial_one_hot = _one_hot_for_sequences(initial, device)
    logits = nn.Parameter(initial_one_hot + 0.05 * torch.randn_like(initial_one_hot))
    parameters: list[nn.Parameter] = [logits]
    if name == "fast_seqprop":
        gamma = nn.Parameter(torch.ones(100, 4, device=device))
        beta = nn.Parameter(torch.zeros(100, 4, device=device))
        parameters.extend([gamma, beta])
    optimizer = torch.optim.Adam(parameters, lr=config.gradient_learning_rate)
    tracker = SaturationTracker(visible_set, config)
    query_count = step = 0
    reason = "emergency_query_ceiling"
    while query_count + 100 <= config.maximum_queries:
        optimizer.zero_grad(set_to_none=True)
        scaled = _fast_seqprop_logits(logits, gamma, beta) if name == "fast_seqprop" else logits
        probabilities = torch.softmax(scaled, dim=-1)
        straight_through, hard = _straight_through_sample(probabilities)
        scores, stds = differentiable(straight_through)
        (-scores.mean()).backward()
        optimizer.step()
        query_count += 100
        step += 1
        sequences = _sequences_for_one_hot(hard)
        score_values, std_values = scores.detach().cpu().numpy(), stds.detach().cpu().numpy()
        tracker.observe_batch(query_count, sequences, score_values, std_values)
        if tracker.plateau_reached(query_count):
            reason = "post_knee_plateau"
            break
        if time.monotonic() - started_at >= config.max_runtime_seconds:
            reason = "runtime_limit"
            break
    return tracker.finalize(name, query_count, reason, {
        "gradient_steps": step, "learning_rate": config.gradient_learning_rate,
        "runtime_seconds": time.monotonic() - started_at,
    })


def adaptive_cbas(
    ensemble: UTR5WeakEnsemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: SaturationConfig,
    seed: int,
    device: torch.device,
) -> SaturatedResult:
    _set_seed(seed)
    started_at = time.monotonic()
    encoded = torch.as_tensor(_sequences_to_encoded(initial).astype(np.int64), device=device)
    prior = AutoregressiveDNA(50).to(device)
    _fit_autoregressive(prior, encoded, epochs=config.cbas_pretrain_epochs, learning_rate=3e-3)
    search = copy.deepcopy(prior)
    surrogate = OnlineSurrogate(ensemble, config, started_at=started_at)
    tracker = SaturationTracker(visible_set, config)
    threshold = -np.inf
    rounds = 0
    reason = "emergency_query_ceiling"
    while surrogate.remaining >= config.cbas_batch_size:
        samples = search.sample(config.cbas_batch_size, device)
        sequences = _encoded_to_sequences(samples.cpu().numpy())
        scores, stds = surrogate.query(sequences)
        rounds += 1
        _add_and_observe(tracker, surrogate, sequences, scores, stds, rounds)
        threshold = max(threshold, float(np.quantile(scores, config.cbas_quantile)))
        uncertainty = np.maximum(stds, 1e-3)
        success = np.clip(ndtr((scores - threshold) / uncertainty), 1e-12, 1.0)
        with torch.no_grad():
            log_weights = np.log(success) + prior.log_prob(samples).cpu().numpy() - search.log_prob(samples).cpu().numpy()
        log_weights = np.clip(log_weights - log_weights.max(), -30, 0)
        weights = torch.as_tensor(np.exp(log_weights), dtype=torch.float32, device=device)
        _fit_autoregressive(
            search, samples, epochs=config.cbas_update_epochs, learning_rate=1e-3,
            weights=weights, batch_size=config.cbas_batch_size,
        )
        if tracker.plateau_reached(surrogate.query_count):
            reason = "post_knee_plateau"
            break
        if surrogate.runtime_limit_reached:
            reason = "runtime_limit"
            break
    return tracker.finalize("cbas", surrogate.query_count, reason, {
        "rounds": rounds, "final_threshold": threshold, "runtime_seconds": surrogate.runtime_seconds,
    })


def adaptive_den(
    differentiable: DifferentiableUTR5Ensemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: SaturationConfig,
    seed: int,
) -> SaturatedResult:
    _set_seed(seed)
    started_at = time.monotonic()
    device = differentiable.label_means.device
    generator = DeepExplorationGenerator(16, 50).to(device)
    noise = torch.randn(100, 16, device=device)
    target = torch.as_tensor(_sequences_to_encoded(initial).astype(np.int64), device=device)
    pretrainer = torch.optim.Adam(generator.parameters(), lr=3e-3)
    for _ in range(config.den_seed_pretrain_epochs):
        loss = F.cross_entropy(generator(noise).reshape(-1, 4), target.reshape(-1))
        pretrainer.zero_grad(set_to_none=True)
        loss.backward()
        pretrainer.step()
    optimizer = torch.optim.Adam(generator.parameters(), lr=3e-3)
    tracker = SaturationTracker(visible_set, config)
    query_count = step = 0
    scale = max(float(differentiable.label_stds.mean().cpu()), 1e-3)
    reason = "emergency_query_ceiling"
    while query_count + 100 <= config.maximum_queries:
        probabilities = torch.softmax(generator(noise), dim=-1)
        straight_through, hard = _straight_through_sample(probabilities)
        scores, stds = differentiable(straight_through)
        similarity = (straight_through[:50] * straight_through[50:]).sum(-1).mean()
        loss = -scores.mean() / scale + config.den_diversity_weight * similarity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        optimizer.step()
        query_count += 100
        step += 1
        sequences = _sequences_for_one_hot(hard)
        score_values, std_values = scores.detach().cpu().numpy(), stds.detach().cpu().numpy()
        tracker.observe_batch(query_count, sequences, score_values, std_values)
        if tracker.plateau_reached(query_count):
            reason = "post_knee_plateau"
            break
        if time.monotonic() - started_at >= config.max_runtime_seconds:
            reason = "runtime_limit"
            break
    return tracker.finalize("den", query_count, reason, {
        "steps": step, "diversity_weight": config.den_diversity_weight,
        "runtime_seconds": time.monotonic() - started_at,
    })


def random_at_budget(
    name: str,
    ensemble: UTR5WeakEnsemble,
    visible_set: set[str],
    budget: int,
    config: SaturationConfig,
    seed: int,
) -> OptimizationResult:
    rng = _set_seed(seed)
    started_at = time.monotonic()
    archive = CandidateArchive()
    queried = 0
    while queried < budget:
        count = min(10_000, budget - queried)
        sequences = _random_sequences(count, rng)
        scores, stds = ensemble.predict(sequences, batch_size=config.prediction_batch_size)
        queried += count
        archive.add(sequences, scores, stds, queried)
        if time.monotonic() - started_at >= config.max_runtime_seconds:
            break
    sequences, scores, stds = archive.ranked(visible_set, 100)
    return OptimizationResult(
        name, sequences, scores, stds, queried,
        {
            "budget_source": name.replace("random_screening_", ""),
            "target_budget": budget,
            "selection": "top 100 weak predictions",
            "runtime_seconds": time.monotonic() - started_at,
            "runtime_truncated": queried < budget,
        },
    )


def _fraction_name(value: float) -> str:
    return f"bottom_{value * 100:g}pct"


def run_saturation_benchmark(
    *,
    checkpoint_root: str | Path = "BioSeqBench/results/utr5_weak_cnns",
    strong_checkpoint: str | Path = "BioSeqBench/checkpoints/strong_test_guided_v1/utr5_mpra/optimus5prime/best.pt",
    data_root: str | Path = "BioSeqBench/dataset/utr5_mpra",
    output_dir: str | Path = "BioSeqBench/results/utr5_optimization_saturation90_all60",
    device: str | torch.device | None = None,
    config: SaturationConfig | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    config = config or SaturationConfig()
    config.validate()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(Path(data_root) / "train.csv", usecols=["sequence", "label"])
    test = pd.read_csv(Path(data_root) / "test.csv", usecols=["sequence", "label"])
    train_sequences = train.sequence.astype(str).to_numpy()
    train_labels = train.label.to_numpy(dtype=float)
    order = np.lexsort((train_sequences, train_labels))
    ranked_sequences = train_sequences[order]
    weak_protocol = json.loads((Path(checkpoint_root) / "protocol.json").read_text())
    split_seed = int(weak_protocol["split_seed"])
    validation_fraction = float(weak_protocol["validation_fraction"])
    measured = dict(zip(train_sequences, train_labels))
    reference = pd.concat([train, test], ignore_index=True)
    oracle = Optimus5PrimeOracle(strong_checkpoint, device=device)
    reference_scores = _load_reference_scores(oracle, reference, output / "strong_reference_predictions.npz", config.prediction_batch_size)
    reference_sorted = np.sort(reference_scores)
    protocol = {
        **asdict(config),
        "seeded_algorithms": list(SEEDED_ALGORITHMS),
        "selection_rule": (
            f"earliest archive reaching baseline + {config.saturation_fraction:g}*"
            "(observed plateau-baseline)"
        ),
        "knee_rule": "maximum normalized vertical distance y_norm-x_norm on monotone top-100 curve",
        "plateau_rule": "no cumulative top-100 mean improvement >= material_improvement for plateau_patience_queries",
        "random_rule": "per model/replicate median and arithmetic mean of eight seeded-algorithm q90 counts",
        "weak_model_policy": "each of the 60 checkpoints is optimized independently; no weak ensemble averaging",
        "initialization_policy": (
            "per weak model, all eight seeded optimizers share the same 50 highest-label sequences "
            "from that checkpoint's actual fit split plus 50 uniform full-space random sequences"
        ),
        "runtime_policy": "soft per-model-per-algorithm limit checked between optimizer iterations",
        "device": str(device),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries: list[dict] = []
    all_candidates: list[pd.DataFrame] = []
    all_initial: list[pd.DataFrame] = []
    convergence_rows: list[dict] = []
    for fraction_index, fraction in enumerate(config.train_fractions):
        model_name = _fraction_name(fraction)
        visible_count = round(len(train) * fraction)
        visible_indices = order[:visible_count]
        fit_local, _ = _split_visible_pool(visible_count, validation_fraction, split_seed)
        training_indices = visible_indices[fit_local]
        training_sequences = train_sequences[training_indices]
        training_set = set(training_sequences.tolist())
        training_digest = _index_digest(training_indices)
        for replicate in range(config.num_replicates):
            ensemble = UTR5WeakEnsemble(
                Path(checkpoint_root) / model_name,
                device=device,
                member_indices=[replicate],
            )
            differentiable = DifferentiableUTR5Ensemble(ensemble).to(device)
            weak_state = ensemble.states[0]
            if weak_state.get("training_indices_sha256") != training_digest:
                raise ValueError(
                    f"Training split mismatch for {model_name} member_{replicate:02d}"
                )
            weak_model_seed = int(weak_state["seed"])
            weak_checkpoint = str(ensemble.checkpoint_paths[0])
            run_dir = output / model_name / f"run_{replicate:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            init_seed = config.base_seed + fraction_index * 100_000 + replicate * 1_000 + 997
            initial, sources = _shared_initial_sequences(
                training_sequences, config, np.random.default_rng(init_seed)
            )
            init_id = hashlib.sha256("".join(initial.tolist()).encode()).hexdigest()[:16]
            weak_initial, weak_std = ensemble.predict(initial, batch_size=config.prediction_batch_size)
            strong_initial = oracle.predict(initial, batch_size=config.prediction_batch_size)
            initial_frame = pd.DataFrame({
                "sequence": initial, "source": sources, "in_weak_train": np.isin(initial, training_sequences),
                "measured_label_if_observed": [measured.get(sequence, np.nan) for sequence in initial],
                "weak_prediction": weak_initial, "weak_uncertainty": weak_std, "strong_score": strong_initial,
                "initialization_seed": init_seed, "initialization_id": init_id,
                "weak_model_fraction": fraction, "replicate": replicate,
                "weak_model_member": replicate, "weak_model_seed": weak_model_seed,
                "weak_checkpoint": weak_checkpoint,
            })
            initial_frame.to_csv(run_dir / "shared_initial_sequences.csv", index=False)
            all_initial.append(initial_frame)
            condition_results: dict[str, OptimizationResult] = {}
            for algorithm_index, algorithm in enumerate(SEEDED_ALGORITHMS):
                candidate_path = run_dir / f"{algorithm}_candidates.csv"
                summary_path = run_dir / f"{algorithm}_summary.json"
                trajectory_path = run_dir / f"{algorithm}_trajectory.csv"
                if resume and candidate_path.exists() and summary_path.exists() and trajectory_path.exists():
                    summary = json.loads(summary_path.read_text())
                    was_truncated = (
                        summary.get("stop_reason") == "maximum_queries"
                        and int(summary.get("total_convergence_queries", 0)) < config.maximum_queries
                    )
                    if not was_truncated:
                        scored = pd.read_csv(candidate_path)
                        all_candidates.append(scored)
                        summaries.append(summary)
                        convergence_rows.append({key: summary[key] for key in (
                            "weak_model_fraction", "replicate", "algorithm", "q90_query_count",
                            "knee_query_count", "total_convergence_queries", "initial_top100_mean",
                            "saturation_top100_mean", "q90_top100_mean", "stop_reason",
                        )})
                        print(f"Skipping {model_name} run={replicate:02d} {algorithm}", flush=True)
                        continue
                    print(
                        f"Extending truncated {model_name} run={replicate:02d} {algorithm} "
                        f"from {summary['total_convergence_queries']:,} to max {config.maximum_queries:,}",
                        flush=True,
                    )
                seed = config.base_seed + fraction_index * 100_000 + replicate * 1_000 + algorithm_index * 101
                print(f"Saturating {model_name} run={replicate:02d} {algorithm}", flush=True)
                if algorithm == "ism_greedy":
                    saturated = adaptive_ism(ensemble, initial, training_set, config, seed)
                elif algorithm == "simulated_annealing":
                    saturated = adaptive_sa(ensemble, initial, training_set, config, seed)
                elif algorithm == "genetic_algorithm":
                    saturated = adaptive_ga(ensemble, initial, training_set, config, seed)
                elif algorithm == "adalead":
                    saturated = adaptive_adalead(ensemble, initial, training_set, config, seed)
                elif algorithm in {"vanilla_activation_maximization", "fast_seqprop"}:
                    saturated = adaptive_gradient(algorithm, differentiable, initial, training_set, config, seed)
                elif algorithm == "cbas":
                    saturated = adaptive_cbas(ensemble, initial, training_set, config, seed, device)
                elif algorithm == "den":
                    saturated = adaptive_den(differentiable, initial, training_set, config, seed)
                else:
                    raise AssertionError(algorithm)
                saturated.trajectory.to_csv(trajectory_path, index=False)
                result = saturated.result
                scored, summary = _evaluate_result(result, oracle, reference_sorted, measured, fraction)
                summary.update({
                    "replicate": replicate, "optimizer_seed": seed, "initialization_seed": init_seed,
                    "initialization_id": init_id, "uses_shared_initialization": True,
                    "weak_model_member": replicate, "weak_model_seed": weak_model_seed,
                    "weak_checkpoint": weak_checkpoint,
                    "q90_query_count": saturated.q90_query_count,
                    "knee_query_count": saturated.knee_query_count,
                    "total_convergence_queries": saturated.total_convergence_queries,
                    "initial_top100_mean": saturated.initial_top100_mean,
                    "saturation_top100_mean": saturated.saturation_top100_mean,
                    "q90_top100_mean": saturated.q90_top100_mean,
                    "stop_reason": saturated.stop_reason,
                    "optimization_runtime_seconds": float(result.metadata.get("runtime_seconds", np.nan)),
                })
                for key, value in {
                    "replicate": replicate, "optimizer_seed": seed, "initialization_seed": init_seed,
                    "initialization_id": init_id, "uses_shared_initialization": True,
                    "weak_model_member": replicate, "weak_model_seed": weak_model_seed,
                    "weak_checkpoint": weak_checkpoint,
                    "q90_query_count": saturated.q90_query_count,
                    "total_convergence_queries": saturated.total_convergence_queries,
                }.items():
                    scored[key] = value
                scored.to_csv(candidate_path, index=False)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                all_candidates.append(scored)
                summaries.append(summary)
                convergence_rows.append({key: summary[key] for key in (
                    "weak_model_fraction", "replicate", "algorithm", "q90_query_count",
                    "knee_query_count", "total_convergence_queries", "initial_top100_mean",
                    "saturation_top100_mean", "q90_top100_mean", "stop_reason",
                )})
                condition_results[algorithm] = result
                print(
                    f"  knee={saturated.knee_query_count:,} q90={saturated.q90_query_count:,} "
                    f"stop={saturated.total_convergence_queries:,} weak_sat={saturated.saturation_top100_mean:.4f}",
                    flush=True,
                )
            condition_convergence = [row for row in convergence_rows if np.isclose(row["weak_model_fraction"], fraction) and row["replicate"] == replicate]
            q90_values = np.asarray([row["q90_query_count"] for row in condition_convergence if row["algorithm"] in SEEDED_ALGORITHMS])
            if len(q90_values) != len(SEEDED_ALGORITHMS):
                raise RuntimeError("missing seeded-algorithm q90 counts for Random budget calculation")
            random_budgets = {
                "random_screening_median_q90": int(round(float(np.median(q90_values)))),
                "random_screening_mean_q90": int(round(float(np.mean(q90_values)))),
            }
            random_seed = config.base_seed + fraction_index * 100_000 + replicate * 1_000 + 909
            for name, budget in random_budgets.items():
                candidate_path = run_dir / f"{name}_candidates.csv"
                summary_path = run_dir / f"{name}_summary.json"
                if resume and candidate_path.exists() and summary_path.exists():
                    previous = json.loads(summary_path.read_text())
                    if int(previous.get("q90_query_count", -1)) == budget:
                        all_candidates.append(pd.read_csv(candidate_path))
                        summaries.append(previous)
                        continue
                result = random_at_budget(name, ensemble, training_set, budget, config, random_seed)
                scored, summary = _evaluate_result(result, oracle, reference_sorted, measured, fraction)
                summary.update({
                    "replicate": replicate, "optimizer_seed": random_seed, "initialization_seed": init_seed,
                    "initialization_id": init_id, "uses_shared_initialization": False,
                    "weak_model_member": replicate, "weak_model_seed": weak_model_seed,
                    "weak_checkpoint": weak_checkpoint,
                    "q90_query_count": result.query_count, "knee_query_count": np.nan,
                    "total_convergence_queries": result.query_count, "initial_top100_mean": np.nan,
                    "saturation_top100_mean": np.nan, "q90_top100_mean": float(result.weak_scores.mean()),
                    "stop_reason": (
                        "runtime_limit" if result.metadata.get("runtime_truncated")
                        else "matched_seeded_optimizer_q90_budget"
                    ),
                    "optimization_runtime_seconds": float(result.metadata.get("runtime_seconds", np.nan)),
                })
                for key, value in {
                    "replicate": replicate, "optimizer_seed": random_seed, "initialization_seed": init_seed,
                    "initialization_id": init_id, "uses_shared_initialization": False,
                    "weak_model_member": replicate, "weak_model_seed": weak_model_seed,
                    "weak_checkpoint": weak_checkpoint,
                    "q90_query_count": result.query_count,
                    "total_convergence_queries": result.query_count,
                }.items():
                    scored[key] = value
                scored.to_csv(candidate_path, index=False)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                all_candidates.append(scored)
                summaries.append(summary)
            top_sequences = training_sequences[-100:][::-1]
            top_scores, top_stds = ensemble.predict(top_sequences, batch_size=config.prediction_batch_size)
            top_result = OptimizationResult("training_top100", top_sequences, top_scores, top_stds, 0, {"reference": "measured Train Top-100"})
            scored, summary = _evaluate_result(top_result, oracle, reference_sorted, measured, fraction)
            summary.update({
                "replicate": replicate, "optimizer_seed": 0, "initialization_seed": init_seed,
                "initialization_id": init_id, "uses_shared_initialization": False,
                "weak_model_member": replicate, "weak_model_seed": weak_model_seed,
                "weak_checkpoint": weak_checkpoint,
                "q90_query_count": 0, "knee_query_count": np.nan, "total_convergence_queries": 0,
                "initial_top100_mean": np.nan, "saturation_top100_mean": np.nan,
                "q90_top100_mean": float(top_scores.mean()), "stop_reason": "observed_reference",
            })
            scored["replicate"] = replicate
            scored["optimizer_seed"] = 0
            scored["initialization_seed"] = init_seed
            scored["initialization_id"] = init_id
            scored["uses_shared_initialization"] = False
            scored["weak_model_member"] = replicate
            scored["weak_model_seed"] = weak_model_seed
            scored["weak_checkpoint"] = weak_checkpoint
            scored["q90_query_count"] = 0
            scored["total_convergence_queries"] = 0
            scored.to_csv(run_dir / "training_top100_candidates.csv", index=False)
            (run_dir / "training_top100_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            all_candidates.append(scored)
            summaries.append(summary)
            del differentiable, ensemble
            torch.cuda.empty_cache() if device.type == "cuda" else None
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "summary.csv", index=False)
    pd.concat(all_candidates, ignore_index=True).to_csv(output / "all_candidates.csv", index=False)
    initial_frame = pd.concat(all_initial, ignore_index=True)
    initial_frame.to_csv(output / "all_shared_initial_sequences.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv(output / "convergence_summary.csv", index=False)
    _aggregate(summary_frame).to_csv(output / "aggregate_summary.csv", index=False)
    return summary_frame
