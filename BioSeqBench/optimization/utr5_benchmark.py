"""Optimus 5'UTR design benchmark over twelve lower-tail weak models.

Unlike TFBind8, the 4^50 design space cannot be enumerated. Weak ensembles are
therefore queried online under a strict common budget, and the frozen full-data
Optimus model is invoked only after candidate generation as the strong oracle.
"""

from __future__ import annotations

import copy
import heapq
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from torch import nn
from torch.nn import functional as F

from BioSeqBench.models.utr5_mpra import (
    DifferentiableUTR5Ensemble,
    UTR5WeakEnsemble,
    encode_utr_sequences,
)
from BioSeqBench.optimization.tfbind8_benchmark import (
    AutoregressiveDNA,
    DeepExplorationGenerator,
    _adalead_parent_mask,
    _fast_seqprop_logits,
    _fit_autoregressive,
    _straight_through_sample,
)
from BioSeqBench.oracles.optimus5prime import Optimus5PrimeOracle


DNA = np.asarray(list("ACGT"))
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
ALGORITHM_NAMES = (
    "training_top100",
    "random_screening",
    "ism_greedy",
    "simulated_annealing",
    "genetic_algorithm",
    "adalead",
    "vanilla_activation_maximization",
    "fast_seqprop",
    "cbas",
    "den",
)


@dataclass(frozen=True)
class UTR5OptimizationConfig:
    train_fractions: tuple[float, ...] = DEFAULT_FRACTIONS
    algorithms: tuple[str, ...] = ALGORITHM_NAMES
    num_candidates: int = 100
    query_budget: int = 10_000
    num_replicates: int = 5
    base_seed: int = 20260826
    top_visible_initializations: int = 50
    random_initializations: int = 50
    prediction_batch_size: int = 2048
    gradient_learning_rate: float = 0.1
    cbas_quantile: float = 0.8
    cbas_rounds: int = 10
    cbas_pretrain_epochs: int = 20
    cbas_update_epochs: int = 10
    den_diversity_weight: float = 0.1
    den_seed_pretrain_epochs: int = 200

    def validate(self) -> None:
        if not self.train_fractions or any(value <= 0 or value > 0.95 for value in self.train_fractions):
            raise ValueError("train_fractions must lie in (0, 0.95]")
        unknown = set(self.algorithms) - set(ALGORITHM_NAMES)
        if unknown:
            raise ValueError(f"Unknown algorithms: {sorted(unknown)}")
        if self.num_candidates < 1 or self.query_budget < self.num_candidates:
            raise ValueError("query_budget must be at least num_candidates")
        if self.query_budget % self.num_candidates:
            raise ValueError("query_budget must be divisible by num_candidates")
        if self.top_visible_initializations + self.random_initializations != self.num_candidates:
            raise ValueError("shared initialization must contain exactly num_candidates sequences")
        if self.num_replicates < 1:
            raise ValueError("num_replicates must be positive")
        if not 0 < self.cbas_quantile < 1 or self.cbas_rounds < 1:
            raise ValueError("invalid CbAS configuration")


@dataclass
class OptimizationResult:
    algorithm: str
    sequences: np.ndarray
    weak_scores: np.ndarray
    weak_stds: np.ndarray
    query_count: int
    metadata: dict[str, float | int | str | bool]


def _set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _fraction_name(fraction: float) -> str:
    return f"bottom_{fraction * 100:g}pct"


def _encoded_to_sequences(encoded: np.ndarray) -> np.ndarray:
    values = np.asarray(encoded, dtype=np.uint8)
    return np.asarray(["".join(row) for row in DNA[values]], dtype=str)


def _sequences_to_encoded(sequences: Sequence[str] | np.ndarray) -> np.ndarray:
    values = np.asarray(sequences, dtype=str).reshape(-1)
    if not len(values):
        return np.empty((0, 50), dtype=np.uint8)
    if np.any(np.char.str_len(values) != 50):
        raise ValueError("all UTR designs must have length 50")
    raw = np.frombuffer("".join(values.tolist()).encode("ascii"), dtype=np.uint8).reshape(-1, 50)
    lookup = np.full(256, -1, dtype=np.int8)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4, dtype=np.int8)
    encoded = lookup[raw]
    if np.any(encoded < 0):
        raise ValueError("UTR designs must contain only A/C/G/T")
    return encoded.astype(np.uint8)


def _random_sequences(count: int, rng: np.random.Generator) -> np.ndarray:
    return _encoded_to_sequences(rng.integers(0, 4, size=(count, 50), dtype=np.uint8))


def _random_single_mutants(sequences: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    encoded = _sequences_to_encoded(sequences)
    positions = rng.integers(0, 50, size=len(encoded))
    offsets = rng.integers(1, 4, size=len(encoded))
    encoded[np.arange(len(encoded)), positions] = (
        encoded[np.arange(len(encoded)), positions] + offsets
    ) % 4
    return _encoded_to_sequences(encoded)


def _all_single_mutants(sequence: str) -> np.ndarray:
    encoded = _sequences_to_encoded([sequence])[0]
    mutants = np.repeat(encoded[None, :], 150, axis=0)
    row = 0
    for position in range(50):
        for base in range(4):
            if base != encoded[position]:
                mutants[row, position] = base
                row += 1
    return _encoded_to_sequences(mutants)


def _one_hot_for_sequences(sequences: np.ndarray, device: torch.device) -> torch.Tensor:
    return encode_utr_sequences(sequences).transpose(1, 2).to(device)


def _sequences_for_one_hot(one_hot: torch.Tensor) -> np.ndarray:
    encoded = one_hot.detach().argmax(dim=-1).cpu().numpy().astype(np.uint8)
    return _encoded_to_sequences(encoded)


class CandidateArchive:
    def __init__(self) -> None:
        self.records: dict[str, tuple[float, float, int]] = {}

    def add(
        self,
        sequences: Sequence[str] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        stds: Sequence[float] | np.ndarray,
        step: int,
    ) -> None:
        for sequence, score, std in zip(sequences, scores, stds):
            key = str(sequence)
            value = (float(score), float(std), int(step))
            if key not in self.records or value[0] > self.records[key][0]:
                self.records[key] = value

    def ranked(self, excluded: set[str], count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eligible = [(sequence, *record) for sequence, record in self.records.items() if sequence not in excluded]
        if len(eligible) < count:
            raise RuntimeError(f"only {len(eligible)} unique novel candidates were queried; {count} required")
        selected = heapq.nsmallest(count, eligible, key=lambda item: (-item[1], item[0]))
        return (
            np.asarray([item[0] for item in selected], dtype=str),
            np.asarray([item[1] for item in selected], dtype=np.float64),
            np.asarray([item[2] for item in selected], dtype=np.float64),
        )


class BudgetedSurrogate:
    def __init__(self, ensemble: UTR5WeakEnsemble, budget: int, batch_size: int) -> None:
        self.ensemble = ensemble
        self.budget = int(budget)
        self.batch_size = int(batch_size)
        self.query_count = 0

    @property
    def remaining(self) -> int:
        return self.budget - self.query_count

    def query(self, sequences: Sequence[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        if len(values) > self.remaining:
            raise RuntimeError(f"surrogate budget exceeded: requested {len(values)}, remaining {self.remaining}")
        scores, stds = self.ensemble.predict(values, batch_size=self.batch_size)
        self.query_count += len(values)
        return scores.astype(np.float64), stds.astype(np.float64)


def _shared_initial_sequences(
    visible_sequences: np.ndarray,
    config: UTR5OptimizationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    top = np.asarray(visible_sequences[-config.top_visible_initializations :][::-1], dtype=str)
    excluded = set(top.tolist())
    random_values: list[str] = []
    while len(random_values) < config.random_initializations:
        for sequence in _random_sequences(config.random_initializations, rng):
            if sequence not in excluded:
                excluded.add(str(sequence))
                random_values.append(str(sequence))
                if len(random_values) == config.random_initializations:
                    break
    initial = np.concatenate([top, np.asarray(random_values, dtype=str)])
    sources = np.asarray(["train_top"] * len(top) + ["full_space_random"] * len(random_values))
    return initial, sources


def _finalize(
    name: str,
    archive: CandidateArchive,
    surrogate: BudgetedSurrogate,
    visible_set: set[str],
    config: UTR5OptimizationConfig,
    **metadata: float | int | str | bool,
) -> OptimizationResult:
    sequences, scores, stds = archive.ranked(visible_set, config.num_candidates)
    return OptimizationResult(name, sequences, scores, stds, surrogate.query_count, dict(metadata))


def optimize_training_top100(
    ensemble: UTR5WeakEnsemble,
    visible_sequences: np.ndarray,
    config: UTR5OptimizationConfig,
) -> OptimizationResult:
    sequences = np.asarray(visible_sequences[-config.num_candidates :][::-1], dtype=str)
    scores, stds = ensemble.predict(sequences, batch_size=config.prediction_batch_size)
    return OptimizationResult(
        "training_top100", sequences, scores, stds, 0,
        {"reference": "highest measured-MRL sequences in visible lower-tail pool", "uses_known_labels": True},
    )


def optimize_random_screening(
    ensemble: UTR5WeakEnsemble,
    visible_set: set[str],
    config: UTR5OptimizationConfig,
    seed: int,
) -> OptimizationResult:
    rng = _set_seed(seed)
    surrogate = BudgetedSurrogate(ensemble, config.query_budget, config.prediction_batch_size)
    sampled = _random_sequences(config.query_budget, rng)
    scores, stds = surrogate.query(sampled)
    archive = CandidateArchive()
    archive.add(sampled, scores, stds, 0)
    return _finalize("random_screening", archive, surrogate, visible_set, config)


def optimize_ism_greedy(
    ensemble: UTR5WeakEnsemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: UTR5OptimizationConfig,
    seed: int,
) -> OptimizationResult:
    rng = _set_seed(seed)
    surrogate = BudgetedSurrogate(ensemble, config.query_budget, config.prediction_batch_size)
    current = initial.copy()
    current_scores, current_stds = surrogate.query(current)
    archive = CandidateArchive()
    archive.add(current, current_scores, current_stds, 0)
    rounds = 0
    while surrogate.remaining >= 150:
        progressed = False
        for index in rng.permutation(len(current)):
            if surrogate.remaining < 150:
                break
            mutants = _all_single_mutants(str(current[index]))
            scores, stds = surrogate.query(mutants)
            rounds += 1
            archive.add(mutants, scores, stds, rounds)
            best = int(np.argmax(scores))
            if scores[best] > current_scores[index]:
                current[index], current_scores[index], current_stds[index] = mutants[best], scores[best], stds[best]
                progressed = True
        if not progressed and surrogate.remaining < len(current) * 150:
            break
    if surrogate.remaining:
        tail = _random_single_mutants(np.resize(current, surrogate.remaining), rng)
        scores, stds = surrogate.query(tail)
        archive.add(tail, scores, stds, rounds + 1)
    return _finalize("ism_greedy", archive, surrogate, visible_set, config, rounds=rounds, neighborhood=150)


def optimize_simulated_annealing(
    ensemble: UTR5WeakEnsemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: UTR5OptimizationConfig,
    seed: int,
) -> OptimizationResult:
    rng = _set_seed(seed)
    surrogate = BudgetedSurrogate(ensemble, config.query_budget, config.prediction_batch_size)
    current = initial.copy()
    current_scores, current_stds = surrogate.query(current)
    archive = CandidateArchive()
    archive.add(current, current_scores, current_stds, 0)
    temperature0 = max(float(np.std(current_scores)), 1e-3)
    steps = 0
    while surrogate.remaining:
        batch = min(len(current), surrogate.remaining)
        proposals = _random_single_mutants(current[:batch], rng)
        scores, stds = surrogate.query(proposals)
        steps += 1
        archive.add(proposals, scores, stds, steps)
        temperature = max(temperature0 * (0.01 ** (surrogate.query_count / config.query_budget)), 1e-6)
        delta = scores - current_scores[:batch]
        accept = (delta >= 0) | (rng.random(batch) < np.exp(np.minimum(delta / temperature, 0)))
        indices = np.flatnonzero(accept)
        current[indices], current_scores[indices], current_stds[indices] = (
            proposals[accept], scores[accept], stds[accept]
        )
    return _finalize(
        "simulated_annealing", archive, surrogate, visible_set, config,
        steps=steps, initial_temperature=temperature0,
    )


def optimize_genetic_algorithm(
    ensemble: UTR5WeakEnsemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: UTR5OptimizationConfig,
    seed: int,
) -> OptimizationResult:
    rng = _set_seed(seed)
    surrogate = BudgetedSurrogate(ensemble, config.query_budget, config.prediction_batch_size)
    population = initial.copy()
    scores, stds = surrogate.query(population)
    archive = CandidateArchive()
    archive.add(population, scores, stds, 0)
    elite_count = max(10, len(population) // 5)
    generation = 0
    while surrogate.remaining:
        count = min(len(population), surrogate.remaining)
        order = np.argsort(scores)[::-1]
        elites = _sequences_to_encoded(population[order[:elite_count]])
        parent_a = elites[rng.integers(0, elite_count, size=count)]
        parent_b = elites[rng.integers(0, elite_count, size=count)]
        children = np.where(rng.random(parent_a.shape) < 0.5, parent_a, parent_b)
        mutation = rng.random(children.shape) < 0.02
        children = np.where(mutation, (children + rng.integers(1, 4, size=children.shape)) % 4, children)
        child_sequences = _encoded_to_sequences(children)
        child_scores, child_stds = surrogate.query(child_sequences)
        generation += 1
        archive.add(child_sequences, child_scores, child_stds, generation)
        combined_sequences = np.concatenate([population, child_sequences])
        combined_scores = np.concatenate([scores, child_scores])
        combined_stds = np.concatenate([stds, child_stds])
        keep = np.argsort(combined_scores)[::-1][: len(population)]
        population, scores, stds = combined_sequences[keep], combined_scores[keep], combined_stds[keep]
    return _finalize(
        "genetic_algorithm", archive, surrogate, visible_set, config,
        generations=generation, elite_count=elite_count, mutation_rate=0.02,
    )


def optimize_adalead(
    ensemble: UTR5WeakEnsemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: UTR5OptimizationConfig,
    seed: int,
) -> OptimizationResult:
    rng = _set_seed(seed)
    surrogate = BudgetedSurrogate(ensemble, config.query_budget, config.prediction_batch_size)
    population = initial.copy()
    population_scores, population_stds = surrogate.query(population)
    archive = CandidateArchive()
    archive.add(population, population_scores, population_stds, 0)
    threshold = 0.05
    round_index = 0
    round_cost = max(len(initial), config.query_budget // 10)
    while surrogate.remaining:
        parents = population[_adalead_parent_mask(population_scores, threshold)]
        parent_scores = population_scores[_adalead_parent_mask(population_scores, threshold)]
        if not len(parents):
            parents, parent_scores = population[[int(np.argmax(population_scores))]], np.asarray([population_scores.max()])
        budget = min(round_cost, surrogate.remaining)
        roots = np.resize(parents, min(len(parents), budget))
        root_scores = np.resize(parent_scores, len(roots))
        nodes, node_roots = roots.copy(), root_scores.copy()
        round_sequences: list[str] = []
        round_scores: list[float] = []
        round_stds: list[float] = []
        consumed = 0
        while consumed < budget:
            batch = min(len(nodes), budget - consumed)
            children = _random_single_mutants(nodes[:batch], rng)
            scores, stds = surrogate.query(children)
            consumed += batch
            round_sequences.extend(children.tolist())
            round_scores.extend(scores.tolist())
            round_stds.extend(stds.tolist())
            continued = scores >= node_roots[:batch]
            if continued.any():
                nodes, node_roots = children[continued], node_roots[:batch][continued]
            else:
                nodes = np.resize(roots, min(len(roots), budget - consumed))
                node_roots = np.resize(root_scores, len(nodes))
        round_index += 1
        sequences = np.asarray(round_sequences, dtype=str)
        scores = np.asarray(round_scores)
        stds = np.asarray(round_stds)
        archive.add(sequences, scores, stds, round_index)
        unique_best: dict[str, float] = {}
        for sequence, score in zip(np.concatenate([population, sequences]), np.concatenate([population_scores, scores])):
            unique_best[str(sequence)] = max(float(score), unique_best.get(str(sequence), -np.inf))
        ranked = sorted(unique_best, key=lambda sequence: (-unique_best[sequence], sequence))[: config.num_candidates]
        population = np.asarray(ranked, dtype=str)
        population_scores = np.asarray([unique_best[sequence] for sequence in ranked])
    return _finalize("adalead", archive, surrogate, visible_set, config, rounds=round_index, threshold=threshold)


def optimize_vanilla_activation_maximization(
    ensemble: UTR5WeakEnsemble,
    differentiable: DifferentiableUTR5Ensemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: UTR5OptimizationConfig,
    seed: int,
) -> OptimizationResult:
    rng = _set_seed(seed)
    device = differentiable.label_means.device
    initial_one_hot = _one_hot_for_sequences(initial, device)
    logits = nn.Parameter(initial_one_hot * 1.5 + 0.05 * torch.randn_like(initial_one_hot))
    optimizer = torch.optim.Adam([logits], lr=config.gradient_learning_rate)
    steps = (config.query_budget - config.num_candidates) // config.num_candidates
    proxy: dict[str, float] = {}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probabilities = torch.softmax(logits, dim=-1)
        predictions, _ = differentiable(probabilities)
        (-predictions.mean()).backward()
        optimizer.step()
        for sequence, value in zip(_sequences_for_one_hot(probabilities), predictions.detach().cpu().numpy()):
            proxy[str(sequence)] = max(float(value), proxy.get(str(sequence), -np.inf))
    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=-1)
        soft_scores = differentiable(probabilities)[0].cpu().numpy()
        attempts = 0
        while len([sequence for sequence in proxy if sequence not in visible_set]) < config.num_candidates:
            sampled = torch.multinomial(probabilities.reshape(-1, 4), 1).reshape(config.num_candidates, 50)
            for sequence, value in zip(_encoded_to_sequences(sampled.cpu().numpy()), soft_scores):
                proxy[str(sequence)] = max(float(value), proxy.get(str(sequence), -np.inf))
            attempts += 1
            if attempts >= 1000:
                break
    eligible = [(sequence, value) for sequence, value in proxy.items() if sequence not in visible_set]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if len(eligible) < config.num_candidates:
        for sequence in _random_sequences(config.num_candidates - len(eligible), rng):
            eligible.append((str(sequence), -np.inf))
    final = np.asarray([item[0] for item in eligible[: config.num_candidates]], dtype=str)
    scores, stds = ensemble.predict(final, batch_size=config.prediction_batch_size)
    return OptimizationResult(
        "vanilla_activation_maximization", final, scores, stds,
        steps * config.num_candidates + config.num_candidates,
        {"gradient_steps": steps, "learning_rate": config.gradient_learning_rate, "pwm_sampling_attempts": attempts},
    )


def optimize_fast_seqprop(
    differentiable: DifferentiableUTR5Ensemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: UTR5OptimizationConfig,
    seed: int,
) -> OptimizationResult:
    _set_seed(seed)
    device = differentiable.label_means.device
    initial_one_hot = _one_hot_for_sequences(initial, device)
    logits = nn.Parameter(initial_one_hot + 0.05 * torch.randn_like(initial_one_hot))
    gamma = nn.Parameter(torch.ones(config.num_candidates, 4, device=device))
    beta = nn.Parameter(torch.zeros(config.num_candidates, 4, device=device))
    optimizer = torch.optim.Adam([logits, gamma, beta], lr=config.gradient_learning_rate)
    archive = CandidateArchive()
    steps = config.query_budget // config.num_candidates
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probabilities = torch.softmax(_fast_seqprop_logits(logits, gamma, beta), dim=-1)
        straight_through, hard = _straight_through_sample(probabilities)
        scores, stds = differentiable(straight_through)
        (-scores.mean()).backward()
        optimizer.step()
        archive.add(
            _sequences_for_one_hot(hard), scores.detach().cpu().numpy(), stds.detach().cpu().numpy(), step
        )
    sequences, scores, stds = archive.ranked(visible_set, config.num_candidates)
    return OptimizationResult(
        "fast_seqprop", sequences, scores, stds, steps * config.num_candidates,
        {"gradient_steps": steps, "learning_rate": config.gradient_learning_rate},
    )


def optimize_cbas(
    ensemble: UTR5WeakEnsemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: UTR5OptimizationConfig,
    seed: int,
    device: torch.device,
) -> OptimizationResult:
    _set_seed(seed)
    initial_encoded = torch.as_tensor(_sequences_to_encoded(initial).astype(np.int64), device=device)
    prior = AutoregressiveDNA(50).to(device)
    _fit_autoregressive(prior, initial_encoded, epochs=config.cbas_pretrain_epochs, learning_rate=3e-3)
    search = copy.deepcopy(prior)
    surrogate = BudgetedSurrogate(ensemble, config.query_budget, config.prediction_batch_size)
    archive = CandidateArchive()
    batch_size = config.query_budget // config.cbas_rounds
    threshold = -np.inf
    rounds = 0
    while surrogate.remaining:
        count = min(batch_size, surrogate.remaining)
        samples = search.sample(count, device)
        sequences = _encoded_to_sequences(samples.cpu().numpy())
        scores, stds = surrogate.query(sequences)
        rounds += 1
        archive.add(sequences, scores, stds, rounds)
        threshold = max(threshold, float(np.quantile(scores, config.cbas_quantile)))
        uncertainty = np.maximum(stds, max(float(np.median(stds)) * 0.1, 1e-3))
        success = np.clip(ndtr((scores - threshold) / uncertainty), 1e-12, 1.0)
        with torch.no_grad():
            log_prior = prior.log_prob(samples).cpu().numpy()
            log_search = search.log_prob(samples).cpu().numpy()
        log_weights = np.clip(np.log(success) + log_prior - log_search, -100, 100)
        log_weights = np.clip(log_weights - log_weights.max(), -30, 0)
        weights = torch.as_tensor(np.exp(log_weights), dtype=torch.float32, device=device)
        _fit_autoregressive(
            search, samples, epochs=config.cbas_update_epochs, learning_rate=1e-3,
            weights=weights, batch_size=min(1024, count),
        )
    sequences, scores, stds = archive.ranked(visible_set, config.num_candidates)
    return OptimizationResult(
        "cbas", sequences, scores, stds, surrogate.query_count,
        {"rounds": rounds, "batch_size": batch_size, "final_threshold": threshold, "generator": "autoregressive_gru"},
    )


def optimize_den(
    differentiable: DifferentiableUTR5Ensemble,
    initial: np.ndarray,
    visible_set: set[str],
    config: UTR5OptimizationConfig,
    seed: int,
) -> OptimizationResult:
    _set_seed(seed)
    device = differentiable.label_means.device
    noise_dim = 16
    generator = DeepExplorationGenerator(noise_dim, 50).to(device)
    fixed_noise = torch.randn(config.num_candidates, noise_dim, device=device)
    target = torch.as_tensor(_sequences_to_encoded(initial).astype(np.int64), device=device)
    seed_optimizer = torch.optim.Adam(generator.parameters(), lr=3e-3)
    for _ in range(config.den_seed_pretrain_epochs):
        logits = generator(fixed_noise)
        loss = F.cross_entropy(logits.reshape(-1, 4), target.reshape(-1))
        seed_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        seed_optimizer.step()
    with torch.no_grad():
        reconstruction = float((generator(fixed_noise).argmax(-1) == target).float().mean().cpu())
    optimizer = torch.optim.Adam(generator.parameters(), lr=3e-3)
    archive = CandidateArchive()
    steps = config.query_budget // config.num_candidates
    score_scale = max(float(differentiable.label_stds.mean().detach().cpu()), 1e-3)
    for step in range(steps):
        logits = generator(fixed_noise)
        probabilities = torch.softmax(logits, dim=-1)
        straight_through, hard = _straight_through_sample(probabilities)
        scores, stds = differentiable(straight_through)
        half = config.num_candidates // 2
        similarity = (straight_through[:half] * straight_through[half : 2 * half]).sum(-1).mean()
        loss = -scores.mean() / score_scale + config.den_diversity_weight * similarity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        optimizer.step()
        archive.add(
            _sequences_for_one_hot(hard), scores.detach().cpu().numpy(), stds.detach().cpu().numpy(), step
        )
    sequences, scores, stds = archive.ranked(visible_set, config.num_candidates)
    return OptimizationResult(
        "den", sequences, scores, stds, steps * config.num_candidates,
        {"steps": steps, "diversity_weight": config.den_diversity_weight, "seed_reconstruction_accuracy": reconstruction},
    )


def _pairwise_hamming(encoded: np.ndarray) -> float:
    if len(encoded) < 2:
        return 0.0
    distances = (encoded[:, None, :] != encoded[None, :, :]).sum(-1)
    return float(distances[np.triu_indices(len(encoded), 1)].mean())


def _load_reference_scores(
    oracle: Optimus5PrimeOracle,
    frame: pd.DataFrame,
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            values = np.asarray(archive["strong_scores"], dtype=np.float64)
        if values.shape == (len(frame),):
            return values
    values = oracle.predict(frame["sequence"].astype(str).to_numpy(), batch_size=batch_size)
    np.savez_compressed(cache_path, strong_scores=values)
    return values


def _evaluate_result(
    result: OptimizationResult,
    oracle: Optimus5PrimeOracle,
    reference_sorted: np.ndarray,
    measured_by_sequence: dict[str, float],
    fraction: float,
) -> tuple[pd.DataFrame, dict[str, float | int | str | bool]]:
    strong = oracle.predict(result.sequences)
    percentile = 100.0 * np.searchsorted(reference_sorted, strong, side="right") / len(reference_sorted)
    measured = np.asarray([measured_by_sequence.get(sequence, np.nan) for sequence in result.sequences])
    encoded = _sequences_to_encoded(result.sequences)
    scored = pd.DataFrame(
        {
            "dataset": "utr5_mpra",
            "sequence": result.sequences,
            "strong_score": strong,
            "strong_reference_percentile": percentile,
            "measured_label_if_observed": measured,
            "weak_prediction": result.weak_scores,
            "weak_uncertainty": result.weak_stds,
            "prediction_error": result.weak_scores - strong,
            "algorithm": result.algorithm,
            "weak_model_fraction": fraction,
            "query_count": result.query_count,
        }
    )
    ordered = np.sort(strong)[::-1]
    summary = {
        "dataset": "utr5_mpra",
        "weak_model_fraction": fraction,
        "algorithm": result.algorithm,
        "candidate_count": len(strong),
        "unique_candidate_count": len(set(result.sequences.tolist())),
        "query_count": result.query_count,
        "mean_strong_score": float(strong.mean()),
        "median_strong_score": float(np.median(strong)),
        "top_10_mean_strong_score": float(ordered[:10].mean()),
        "best_strong_score": float(ordered[0]),
        "mean_reference_percentile": float(percentile.mean()),
        "top_5pct_count": int((percentile >= 95).sum()),
        "top_1pct_count": int((percentile >= 99).sum()),
        "top_0_1pct_count": int((percentile >= 99.9).sum()),
        "mean_pairwise_hamming": _pairwise_hamming(encoded),
        "mean_weak_prediction": float(np.mean(result.weak_scores)),
        "mean_absolute_prediction_error": float(np.mean(np.abs(result.weak_scores - strong))),
        "observed_training_sequence_count": int(np.isfinite(measured).sum()),
        "optimizer_metadata": json.dumps(result.metadata, sort_keys=True),
    }
    return scored, summary


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "mean_strong_score", "median_strong_score", "top_10_mean_strong_score", "best_strong_score",
        "top_5pct_count", "top_1pct_count", "top_0_1pct_count", "mean_pairwise_hamming",
        "mean_absolute_prediction_error", "mean_reference_percentile",
    )
    named: dict[str, tuple[str, str]] = {"replicates": ("replicate", "nunique")}
    for metric in metrics:
        named[f"{metric}_mean"] = (metric, "mean")
        named[f"{metric}_std"] = (metric, "std")
        named[f"{metric}_sem"] = (metric, "sem")
    return summary.groupby(["dataset", "weak_model_fraction", "algorithm"], sort=False).agg(**named).reset_index()


def run_utr5_benchmark(
    *,
    checkpoint_root: str | Path = "BioSeqBench/results/utr5_weak_cnns",
    strong_checkpoint: str | Path = "BioSeqBench/checkpoints/strong_test_guided_v1/utr5_mpra/optimus5prime/best.pt",
    data_root: str | Path = "BioSeqBench/dataset/utr5_mpra",
    output_dir: str | Path = "BioSeqBench/results/utr5_optimization_benchmark",
    device: str | torch.device | None = None,
    config: UTR5OptimizationConfig | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    config = config or UTR5OptimizationConfig()
    config.validate()
    compute_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(Path(data_root) / "train.csv", usecols=["sequence", "label"])
    test = pd.read_csv(Path(data_root) / "test.csv", usecols=["sequence", "label"])
    reference = pd.concat([train, test], ignore_index=True)
    train_sequences = train["sequence"].astype(str).to_numpy()
    train_labels = train["label"].to_numpy(dtype=np.float64)
    ranked_indices = np.lexsort((train_sequences, train_labels))
    ranked_sequences = train_sequences[ranked_indices]
    measured_by_sequence = dict(zip(train_sequences, train_labels))
    oracle = Optimus5PrimeOracle(strong_checkpoint, device=compute_device)
    reference_scores = _load_reference_scores(
        oracle, reference, output_path / "strong_reference_predictions.npz", config.prediction_batch_size
    )
    reference_sorted = np.sort(reference_scores)
    protocol = {
        **asdict(config),
        "algorithms": list(config.algorithms),
        "train_fractions": list(config.train_fractions),
        "initialization_policy": "shared 50 measured Train-TOP + 50 uniform full-space random 50-mers per model/replicate",
        "weak_query_policy": f"online Optimus weak-model calls; exactly {config.query_budget} per optimizer",
        "evaluation_policy": "frozen full-data Optimus strong model invoked only after candidate generation",
        "strong_reference_percentiles": "strong predictions for the canonical 280k Sample et al. library",
        "device": str(compute_device),
    }
    (output_path / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries: list[dict] = []
    candidates: list[pd.DataFrame] = []
    initializations: list[pd.DataFrame] = []
    for fraction_index, fraction in enumerate(config.train_fractions):
        name = _fraction_name(fraction)
        model_output = output_path / name
        model_output.mkdir(parents=True, exist_ok=True)
        ensemble = UTR5WeakEnsemble(Path(checkpoint_root) / name, device=compute_device)
        differentiable = DifferentiableUTR5Ensemble(ensemble).to(compute_device)
        visible_count = int(round(len(train) * fraction))
        visible_sequences = ranked_sequences[:visible_count]
        visible_set = set(visible_sequences.tolist())
        for replicate in range(config.num_replicates):
            run_output = model_output / f"run_{replicate:02d}"
            run_output.mkdir(parents=True, exist_ok=True)
            initialization_seed = config.base_seed + fraction_index * 100_000 + replicate * 1_000 + 997
            initial, sources = _shared_initial_sequences(
                visible_sequences, config, np.random.default_rng(initialization_seed)
            )
            initialization_id = hashlib.sha256("".join(initial.tolist()).encode()).hexdigest()[:16]
            weak_initial, weak_initial_std = ensemble.predict(initial, batch_size=config.prediction_batch_size)
            strong_initial = oracle.predict(initial, batch_size=config.prediction_batch_size)
            initial_frame = pd.DataFrame(
                {
                    "sequence": initial,
                    "source": sources,
                    "in_weak_train": np.isin(initial, visible_sequences),
                    "measured_label_if_observed": [measured_by_sequence.get(sequence, np.nan) for sequence in initial],
                    "weak_prediction": weak_initial,
                    "weak_uncertainty": weak_initial_std,
                    "strong_score": strong_initial,
                    "initialization_seed": initialization_seed,
                    "initialization_id": initialization_id,
                    "weak_model_fraction": fraction,
                    "replicate": replicate,
                }
            )
            initial_frame.to_csv(run_output / "shared_initial_sequences.csv", index=False)
            initializations.append(initial_frame)
            for algorithm_index, algorithm in enumerate(config.algorithms):
                algorithm_seed = config.base_seed + fraction_index * 100_000 + replicate * 1_000 + algorithm_index * 101
                candidate_path = run_output / f"{algorithm}_candidates.csv"
                summary_path = run_output / f"{algorithm}_summary.json"
                if resume and candidate_path.exists() and summary_path.exists():
                    candidates.append(pd.read_csv(candidate_path))
                    summaries.append(json.loads(summary_path.read_text()))
                    print(f"Skipping {name} run={replicate:02d} {algorithm}", flush=True)
                    continue
                print(f"Optimizing {name} run={replicate:02d} with {algorithm}", flush=True)
                if algorithm == "training_top100":
                    result = optimize_training_top100(ensemble, visible_sequences, config)
                elif algorithm == "random_screening":
                    result = optimize_random_screening(ensemble, visible_set, config, algorithm_seed)
                elif algorithm == "ism_greedy":
                    result = optimize_ism_greedy(ensemble, initial, visible_set, config, algorithm_seed)
                elif algorithm == "simulated_annealing":
                    result = optimize_simulated_annealing(ensemble, initial, visible_set, config, algorithm_seed)
                elif algorithm == "genetic_algorithm":
                    result = optimize_genetic_algorithm(ensemble, initial, visible_set, config, algorithm_seed)
                elif algorithm == "adalead":
                    result = optimize_adalead(ensemble, initial, visible_set, config, algorithm_seed)
                elif algorithm == "vanilla_activation_maximization":
                    result = optimize_vanilla_activation_maximization(
                        ensemble, differentiable, initial, visible_set, config, algorithm_seed
                    )
                elif algorithm == "fast_seqprop":
                    result = optimize_fast_seqprop(differentiable, initial, visible_set, config, algorithm_seed)
                elif algorithm == "cbas":
                    result = optimize_cbas(ensemble, initial, visible_set, config, algorithm_seed, compute_device)
                elif algorithm == "den":
                    result = optimize_den(differentiable, initial, visible_set, config, algorithm_seed)
                else:
                    raise AssertionError(algorithm)
                expected = 0 if algorithm == "training_top100" else config.query_budget
                if result.query_count != expected:
                    raise RuntimeError(f"{algorithm} used {result.query_count}, expected {expected} weak queries")
                if len(result.sequences) != config.num_candidates or len(set(result.sequences.tolist())) != config.num_candidates:
                    raise RuntimeError(f"{algorithm} did not return exactly {config.num_candidates} unique designs")
                overlap = sum(sequence in visible_set for sequence in result.sequences)
                if algorithm == "training_top100" and overlap != config.num_candidates:
                    raise RuntimeError("training_top100 contains unobserved sequences")
                if algorithm != "training_top100" and overlap:
                    raise RuntimeError(f"{algorithm} returned {overlap} visible training sequences")
                scored, summary = _evaluate_result(
                    result, oracle, reference_sorted, measured_by_sequence, fraction
                )
                for target in (scored,):
                    target["replicate"] = replicate
                    target["optimizer_seed"] = algorithm_seed
                    target["initialization_seed"] = initialization_seed
                    target["initialization_id"] = initialization_id
                    target["uses_shared_initialization"] = algorithm not in {"training_top100", "random_screening"}
                summary.update(
                    {
                        "replicate": replicate,
                        "optimizer_seed": algorithm_seed,
                        "initialization_seed": initialization_seed,
                        "initialization_id": initialization_id,
                        "uses_shared_initialization": algorithm not in {"training_top100", "random_screening"},
                    }
                )
                scored.to_csv(candidate_path, index=False)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                candidates.append(scored)
                summaries.append(summary)
                print(
                    f"  strong max={summary['best_strong_score']:.4f}, mean={summary['mean_strong_score']:.4f}, "
                    f"top1%={summary['top_1pct_count']}", flush=True,
                )
        del differentiable, ensemble
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_path / "summary.csv", index=False)
    pd.concat(candidates, ignore_index=True).to_csv(output_path / "all_candidates.csv", index=False)
    initialization_frame = pd.concat(initializations, ignore_index=True)
    initialization_frame.to_csv(output_path / "all_shared_initial_sequences.csv", index=False)
    initialization_frame.groupby(["weak_model_fraction", "replicate", "source"], sort=False).agg(
        sequence_count=("sequence", "size"),
        mean_strong_score=("strong_score", "mean"),
        best_strong_score=("strong_score", "max"),
        mean_weak_prediction=("weak_prediction", "mean"),
        in_weak_train_fraction=("in_weak_train", "mean"),
        initialization_id=("initialization_id", "first"),
    ).reset_index().to_csv(output_path / "shared_initialization_summary.csv", index=False)
    _aggregate(summary_frame).to_csv(output_path / "aggregate_summary.csv", index=False)
    return summary_frame
