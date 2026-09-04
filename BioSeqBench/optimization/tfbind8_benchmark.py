"""Unified TFBind8 optimization benchmark over fixed lower-tail CNN ensembles.

The benchmark deliberately separates the weak surrogate used during optimization
from the exhaustive experimental landscape used once, after candidate generation.
Every optimizer returns exactly ``num_candidates`` unique sequences outside the
weak model's visible lower-tail pool and is charged the same number of weak-model
evaluations.  The implementation covers black-box local/population methods,
gradient-based sequence design, adaptive conditional sampling, and a diversity-
regularized deep exploration generator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from torch import nn
from torch.nn import functional as F

from BioSeqBench.evaluation.evaluate_tfbind_candidates import evaluate_candidates
from BioSeqBench.models.tfbind8 import TFBindWeakEnsemble
from BioSeqBench.oracles.tfbind import (
    TFBindLandscape,
    codes_to_encoded,
    encoded_to_codes,
    load_tfbind_landscape,
)


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
class TFBind8OptimizationConfig:
    """Shared protocol and algorithm hyperparameters."""

    train_fractions: tuple[float, ...] = (0.01, 0.50, 0.95)
    algorithms: tuple[str, ...] = ALGORITHM_NAMES
    num_candidates: int = 100
    query_budget: int = 10_000
    base_seed: int = 20260825
    top_visible_initializations: int = 50
    random_initializations: int = 50
    gradient_learning_rate: float = 0.1
    den_diversity_weight: float = 0.1
    cbas_quantile: float = 0.8
    cbas_rounds: int = 10
    cbas_pretrain_epochs: int = 20
    cbas_update_epochs: int = 10
    prediction_batch_size: int = 4096
    num_replicates: int = 1
    den_seed_pretrain_epochs: int = 200

    def validate(self) -> None:
        if not self.train_fractions:
            raise ValueError("At least one weak-model fraction is required")
        if any(value <= 0 or value > 0.95 for value in self.train_fractions):
            raise ValueError("train_fractions must lie in (0, 0.95]")
        unknown = set(self.algorithms) - set(ALGORITHM_NAMES)
        if unknown:
            raise ValueError(f"Unknown algorithms: {sorted(unknown)}")
        if self.num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if self.query_budget < self.num_candidates:
            raise ValueError("query_budget must be at least num_candidates")
        if self.query_budget % self.num_candidates:
            raise ValueError("query_budget must be divisible by num_candidates")
        if self.top_visible_initializations + self.random_initializations != self.num_candidates:
            raise ValueError("The shared initialization pool must contain exactly num_candidates sequences")
        if not 0 < self.cbas_quantile < 1:
            raise ValueError("cbas_quantile must lie in (0, 1)")
        if self.cbas_rounds < 1:
            raise ValueError("cbas_rounds must be positive")
        if self.num_replicates < 1:
            raise ValueError("num_replicates must be positive")
        if self.den_seed_pretrain_epochs < 1:
            raise ValueError("den_seed_pretrain_epochs must be positive")


@dataclass
class OptimizationResult:
    algorithm: str
    codes: np.ndarray
    weak_scores: np.ndarray
    weak_stds: np.ndarray
    query_count: int
    metadata: dict[str, float | int | str | bool]


class CandidateArchive:
    """Keep the best recorded surrogate value for every queried discrete design."""

    def __init__(self) -> None:
        self._records: dict[int, tuple[float, float, int]] = {}

    def add(
        self,
        codes: Sequence[int] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        stds: Sequence[float] | np.ndarray,
        step: int,
    ) -> None:
        for code, score, std in zip(codes, scores, stds):
            key = int(code)
            value = (float(score), float(std), int(step))
            previous = self._records.get(key)
            if previous is None or value[0] > previous[0]:
                self._records[key] = value

    def __len__(self) -> int:
        return len(self._records)

    def ranked(
        self,
        excluded_codes: set[int],
        count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eligible = [
            (code, *record)
            for code, record in self._records.items()
            if code not in excluded_codes
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        if len(eligible) < count:
            raise RuntimeError(
                f"Only {len(eligible)} unique novel candidates were queried; {count} are required"
            )
        selected = eligible[:count]
        return (
            np.asarray([item[0] for item in selected], dtype=np.int64),
            np.asarray([item[1] for item in selected], dtype=np.float64),
            np.asarray([item[2] for item in selected], dtype=np.float64),
        )


class BudgetedLookupSurrogate:
    """Budget accounting around precomputed ensemble predictions."""

    def __init__(self, scores: np.ndarray, stds: np.ndarray, budget: int) -> None:
        self.scores = np.asarray(scores, dtype=np.float64)
        self.stds = np.asarray(stds, dtype=np.float64)
        self.budget = int(budget)
        self.query_count = 0

    @property
    def remaining(self) -> int:
        return self.budget - self.query_count

    def query(self, codes: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(codes, dtype=np.int64).reshape(-1)
        if len(values) > self.remaining:
            raise RuntimeError(
                f"Surrogate query budget exceeded: requested {len(values)}, remaining {self.remaining}"
            )
        if np.any(values < 0) or np.any(values >= len(self.scores)):
            raise ValueError("Candidate code outside the TFBind8 landscape")
        self.query_count += len(values)
        return self.scores[values], self.stds[values]


class DifferentiableTFBindEnsemble(nn.Module):
    """Frozen, calibrated ensemble retaining gradients with respect to one-hot input."""

    def __init__(self, ensemble: TFBindWeakEnsemble) -> None:
        super().__init__()
        self.models = nn.ModuleList(ensemble.models)
        self.register_buffer("label_means", torch.tensor(ensemble.label_means, dtype=torch.float32))
        self.register_buffer("label_stds", torch.tensor(ensemble.label_stds, dtype=torch.float32))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def member_predictions(self, one_hot: torch.Tensor) -> torch.Tensor:
        if one_hot.ndim != 3:
            raise ValueError("one_hot must have shape [batch, length, 4] or [batch, 4, length]")
        features = one_hot.transpose(1, 2) if one_hot.shape[-1] == 4 else one_hot
        if features.shape[1] != 4:
            raise ValueError("one_hot must contain four DNA channels")
        predictions = []
        for index, model in enumerate(self.models):
            normalized = model(features).squeeze(-1)
            predictions.append(normalized * self.label_stds[index] + self.label_means[index])
        return torch.stack(predictions)

    def forward(self, one_hot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        members = self.member_predictions(one_hot)
        return members.mean(dim=0), members.std(dim=0, unbiased=False)


def _set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _fraction_name(fraction: float) -> str:
    return f"bottom_{fraction * 100:g}pct"


def _mutate_codes(codes: np.ndarray, positions: np.ndarray, bases: np.ndarray, length: int) -> np.ndarray:
    encoded = codes_to_encoded(np.asarray(codes, dtype=np.int64), length)
    encoded[np.arange(len(encoded)), positions] = bases
    return encoded_to_codes(encoded)


def _random_single_mutants(codes: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    encoded = codes_to_encoded(np.asarray(codes, dtype=np.int64), length)
    positions = rng.integers(0, length, size=len(encoded))
    offsets = rng.integers(1, 4, size=len(encoded))
    encoded[np.arange(len(encoded)), positions] = (
        encoded[np.arange(len(encoded)), positions] + offsets
    ) % 4
    return encoded_to_codes(encoded)


def _all_single_mutants(code: int, length: int) -> np.ndarray:
    encoded = codes_to_encoded(np.asarray([code]), length)[0]
    mutants = np.repeat(encoded[None, :], 3 * length, axis=0)
    row = 0
    for position in range(length):
        for base in range(4):
            if base != encoded[position]:
                mutants[row, position] = base
                row += 1
    return encoded_to_codes(mutants)


def _shared_initial_codes(
    ranked_codes: np.ndarray,
    visible_count: int,
    landscape_size: int,
    config: TFBind8OptimizationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build shared seeds without conditioning random draws on training membership."""
    top_count = min(config.top_visible_initializations, visible_count)
    top_visible = ranked_codes[visible_count - top_count : visible_count][::-1]
    random_count = config.random_initializations
    # Exclude only the selected top-visible seeds to guarantee uniqueness. The
    # random pool otherwise spans the complete 4^L sequence space and can
    # contain either observed or unobserved sequences.
    random_pool = np.setdiff1d(
        np.arange(landscape_size, dtype=np.int64),
        np.asarray(top_visible, dtype=np.int64),
        assume_unique=False,
    )
    random_codes = rng.choice(random_pool, size=random_count, replace=False)
    initial = np.concatenate([top_visible, random_codes]).astype(np.int64)
    sources = np.asarray(["train_top"] * len(top_visible) + ["full_space_random"] * len(random_codes))
    if len(initial) != config.num_candidates or len(np.unique(initial)) != config.num_candidates:
        raise RuntimeError("Shared initialization did not produce num_candidates unique sequences")
    return initial, sources


def _resolve_initial_codes(
    initial_codes: np.ndarray | None,
    ranked_codes: np.ndarray,
    visible_count: int,
    landscape_size: int,
    config: TFBind8OptimizationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if initial_codes is None:
        initial_codes = _shared_initial_codes(
            ranked_codes, visible_count, landscape_size, config, rng
        )[0]
    resolved = np.asarray(initial_codes, dtype=np.int64)
    if len(resolved) != config.num_candidates or len(np.unique(resolved)) != config.num_candidates:
        raise ValueError("initial_codes must contain exactly num_candidates unique sequences")
    return resolved.copy()


def _finalize_black_box(
    name: str,
    archive: CandidateArchive,
    surrogate: BudgetedLookupSurrogate,
    visible_set: set[int],
    config: TFBind8OptimizationConfig,
    **metadata: float | int | str | bool,
) -> OptimizationResult:
    codes, scores, stds = archive.ranked(visible_set, config.num_candidates)
    return OptimizationResult(name, codes, scores, stds, surrogate.query_count, dict(metadata))


def optimize_random_screening(
    scores: np.ndarray,
    stds: np.ndarray,
    ranked_codes: np.ndarray,
    visible_count: int,
    length: int,
    config: TFBind8OptimizationConfig,
    seed: int,
    initial_codes: np.ndarray | None = None,
) -> OptimizationResult:
    del length, initial_codes
    rng = _set_seed(seed)
    surrogate = BudgetedLookupSurrogate(scores, stds, config.query_budget)
    visible_set = set(map(int, ranked_codes[:visible_count]))
    available = np.asarray([code for code in range(len(scores)) if code not in visible_set], dtype=np.int64)
    replace = config.query_budget > len(available)
    sampled = rng.choice(available, size=config.query_budget, replace=replace)
    predicted, uncertainty = surrogate.query(sampled)
    archive = CandidateArchive()
    archive.add(sampled, predicted, uncertainty, step=0)
    return _finalize_black_box(
        "random_screening", archive, surrogate, visible_set, config, sampled_with_replacement=replace
    )


def optimize_training_top100(
    scores: np.ndarray,
    stds: np.ndarray,
    ranked_codes: np.ndarray,
    visible_count: int,
    config: TFBind8OptimizationConfig,
) -> OptimizationResult:
    """Return the highest-activity observed training sequences as a reference."""
    if visible_count < config.num_candidates:
        raise ValueError("The visible training pool contains fewer than num_candidates sequences")
    codes = np.asarray(
        ranked_codes[visible_count - config.num_candidates : visible_count][::-1],
        dtype=np.int64,
    )
    return OptimizationResult(
        "training_top100",
        codes,
        np.asarray(scores[codes], dtype=np.float64),
        np.asarray(stds[codes], dtype=np.float64),
        0,
        {
            "reference": "100 highest-activity sequences in the visible weak-model training pool",
            "uses_known_training_labels": True,
        },
    )


def optimize_ism_greedy(
    scores: np.ndarray,
    stds: np.ndarray,
    ranked_codes: np.ndarray,
    visible_count: int,
    length: int,
    config: TFBind8OptimizationConfig,
    seed: int,
    initial_codes: np.ndarray | None = None,
) -> OptimizationResult:
    rng = _set_seed(seed)
    surrogate = BudgetedLookupSurrogate(scores, stds, config.query_budget)
    visible_set = set(map(int, ranked_codes[:visible_count]))
    current = _resolve_initial_codes(
        initial_codes, ranked_codes, visible_count, len(scores), config, rng
    )
    current_scores, current_stds = surrogate.query(current)
    archive = CandidateArchive()
    archive.add(current, current_scores, current_stds, step=0)
    rounds = 0
    while surrogate.remaining >= 3 * length:
        progressed = False
        order = rng.permutation(len(current))
        for index in order:
            if surrogate.remaining < 3 * length:
                break
            mutants = _all_single_mutants(int(current[index]), length)
            mutant_scores, mutant_stds = surrogate.query(mutants)
            rounds += 1
            archive.add(mutants, mutant_scores, mutant_stds, step=rounds)
            best = int(np.argmax(mutant_scores))
            if mutant_scores[best] > current_scores[index]:
                current[index] = mutants[best]
                current_scores[index] = mutant_scores[best]
                current_stds[index] = mutant_stds[best]
                progressed = True
        if not progressed and surrogate.remaining < len(current) * 3 * length:
            break
    # Consume the otherwise unusable remainder (<3L) with valid one-step
    # proposals so all optimizers receive exactly the advertised budget.
    if surrogate.remaining:
        tail_parents = np.resize(current, surrogate.remaining)
        tail = _random_single_mutants(tail_parents, length, rng)
        tail_scores, tail_stds = surrogate.query(tail)
        archive.add(tail, tail_scores, tail_stds, step=rounds + 1)
    return _finalize_black_box(
        "ism_greedy", archive, surrogate, visible_set, config, mutation_neighborhood=3 * length, rounds=rounds
    )


def optimize_simulated_annealing(
    scores: np.ndarray,
    stds: np.ndarray,
    ranked_codes: np.ndarray,
    visible_count: int,
    length: int,
    config: TFBind8OptimizationConfig,
    seed: int,
    initial_codes: np.ndarray | None = None,
) -> OptimizationResult:
    rng = _set_seed(seed)
    surrogate = BudgetedLookupSurrogate(scores, stds, config.query_budget)
    visible_set = set(map(int, ranked_codes[:visible_count]))
    current = _resolve_initial_codes(
        initial_codes, ranked_codes, visible_count, len(scores), config, rng
    )
    current_scores, current_stds = surrogate.query(current)
    archive = CandidateArchive()
    archive.add(current, current_scores, current_stds, step=0)
    initial_temperature = max(float(np.std(current_scores)), 1e-3)
    steps = 0
    while surrogate.remaining > 0:
        batch_size = min(len(current), surrogate.remaining)
        indices = np.arange(batch_size)
        proposals = _random_single_mutants(current[indices], length, rng)
        proposal_scores, proposal_stds = surrogate.query(proposals)
        steps += 1
        archive.add(proposals, proposal_scores, proposal_stds, step=steps)
        progress = surrogate.query_count / config.query_budget
        temperature = max(initial_temperature * (0.01 ** progress), 1e-6)
        delta = proposal_scores - current_scores[indices]
        accept = (delta >= 0) | (rng.random(batch_size) < np.exp(np.minimum(delta / temperature, 0)))
        accepted = indices[accept]
        current[accepted] = proposals[accept]
        current_scores[accepted] = proposal_scores[accept]
        current_stds[accepted] = proposal_stds[accept]
    return _finalize_black_box(
        "simulated_annealing",
        archive,
        surrogate,
        visible_set,
        config,
        initial_temperature=initial_temperature,
        steps=steps,
    )


def _crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if parent_a.shape != parent_b.shape:
        raise ValueError("Parent arrays must share a shape")
    mask = rng.random(parent_a.shape) < 0.5
    return np.where(mask, parent_a, parent_b)


def optimize_genetic_algorithm(
    scores: np.ndarray,
    stds: np.ndarray,
    ranked_codes: np.ndarray,
    visible_count: int,
    length: int,
    config: TFBind8OptimizationConfig,
    seed: int,
    initial_codes: np.ndarray | None = None,
) -> OptimizationResult:
    rng = _set_seed(seed)
    surrogate = BudgetedLookupSurrogate(scores, stds, config.query_budget)
    visible_set = set(map(int, ranked_codes[:visible_count]))
    population = _resolve_initial_codes(
        initial_codes, ranked_codes, visible_count, len(scores), config, rng
    )
    population_size = len(population)
    population_scores, population_stds = surrogate.query(population)
    archive = CandidateArchive()
    archive.add(population, population_scores, population_stds, step=0)
    generation = 0
    elite_count = max(10, population_size // 5)
    while surrogate.remaining > 0:
        offspring_count = min(population_size, surrogate.remaining)
        order = np.argsort(population_scores)[::-1]
        elites = population[order[:elite_count]]
        elite_encoded = codes_to_encoded(elites, length)
        parent_a = elite_encoded[rng.integers(0, len(elites), size=offspring_count)]
        parent_b = elite_encoded[rng.integers(0, len(elites), size=offspring_count)]
        children = _crossover(parent_a, parent_b, rng)
        mutation_mask = rng.random(children.shape) < (1.0 / length)
        mutation_offsets = rng.integers(1, 4, size=children.shape)
        children = np.where(mutation_mask, (children + mutation_offsets) % 4, children)
        child_codes = encoded_to_codes(children)
        child_scores, child_stds = surrogate.query(child_codes)
        generation += 1
        archive.add(child_codes, child_scores, child_stds, step=generation)
        combined_codes = np.concatenate([population, child_codes])
        combined_scores = np.concatenate([population_scores, child_scores])
        combined_stds = np.concatenate([population_stds, child_stds])
        keep = np.argsort(combined_scores)[::-1][:population_size]
        population = combined_codes[keep]
        population_scores = combined_scores[keep]
        population_stds = combined_stds[keep]
    return _finalize_black_box(
        "genetic_algorithm",
        archive,
        surrogate,
        visible_set,
        config,
        population_size=population_size,
        elite_count=elite_count,
        generations=generation,
        mutation_rate=1.0 / length,
    )


def _adalead_parent_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    maximum = float(values.max())
    cutoff = maximum * (1 - np.sign(maximum) * threshold)
    return values >= cutoff


def optimize_adalead(
    scores: np.ndarray,
    stds: np.ndarray,
    ranked_codes: np.ndarray,
    visible_count: int,
    length: int,
    config: TFBind8OptimizationConfig,
    seed: int,
    initial_codes: np.ndarray | None = None,
) -> OptimizationResult:
    rng = _set_seed(seed)
    surrogate = BudgetedLookupSurrogate(scores, stds, config.query_budget)
    visible_set = set(map(int, ranked_codes[:visible_count]))
    initial = _resolve_initial_codes(
        initial_codes, ranked_codes, visible_count, len(scores), config, rng
    )
    initial_scores, initial_stds = surrogate.query(initial)
    archive = CandidateArchive()
    archive.add(initial, initial_scores, initial_stds, step=0)
    population = initial.copy()
    population_scores = initial_scores.copy()
    threshold = 0.05
    round_index = 0
    target_round_cost = max(len(initial), config.query_budget // 10)
    unobserved_fraction = max(1.0 - visible_count / len(scores), 1e-6)
    novelty_reserve = min(
        surrogate.remaining,
        int(math.ceil(1.6 * config.num_candidates / unobserved_fraction)),
    )
    while surrogate.remaining > novelty_reserve:
        parent_mask = _adalead_parent_mask(population_scores, threshold)
        parents = population[parent_mask]
        parent_scores = population_scores[parent_mask]
        if not len(parents):
            parents = population[np.asarray([int(np.argmax(population_scores))])]
            parent_scores = np.asarray([population_scores.max()])
        round_codes: list[int] = []
        round_scores: list[float] = []
        round_stds: list[float] = []
        round_budget = min(target_round_cost, surrogate.remaining - novelty_reserve)
        roots = np.resize(parents, min(len(parents), round_budget))
        root_scores = np.resize(parent_scores, len(roots))
        nodes = roots.copy()
        node_root_scores = root_scores.copy()
        consumed = 0
        while consumed < round_budget:
            batch_size = min(
                len(nodes),
                round_budget - consumed,
                surrogate.remaining - novelty_reserve,
            )
            if batch_size == 0:
                break
            children = _random_single_mutants(nodes[:batch_size], length, rng)
            child_scores, child_stds = surrogate.query(children)
            consumed += batch_size
            round_codes.extend(map(int, children))
            round_scores.extend(map(float, child_scores))
            round_stds.extend(map(float, child_stds))
            continue_mask = child_scores >= node_root_scores[:batch_size]
            continued = children[continue_mask]
            continued_roots = node_root_scores[:batch_size][continue_mask]
            if len(continued):
                nodes = continued
                node_root_scores = continued_roots
            else:
                nodes = np.resize(roots, min(len(roots), round_budget - consumed))
                node_root_scores = np.resize(root_scores, len(nodes))
        round_index += 1
        if round_codes:
            code_array = np.asarray(round_codes, dtype=np.int64)
            score_array = np.asarray(round_scores)
            std_array = np.asarray(round_stds)
            archive.add(code_array, score_array, std_array, step=round_index)
            combined_codes = np.concatenate([population, code_array])
            combined_scores = np.concatenate([population_scores, score_array])
            unique_best: dict[int, float] = {}
            for code, value in zip(combined_codes, combined_scores):
                unique_best[int(code)] = max(float(value), unique_best.get(int(code), -np.inf))
            ranked = sorted(unique_best, key=lambda code: (-unique_best[code], code))[: max(100, len(initial))]
            population = np.asarray(ranked, dtype=np.int64)
            population_scores = np.asarray([unique_best[code] for code in ranked])
    # Do not condition these proposals on training membership. The reserve
    # simply ensures sufficient full-space coverage when most shared random
    # seeds fall inside a large lower-tail training pool.
    if surrogate.remaining:
        completion_codes = rng.choice(
            np.arange(len(scores), dtype=np.int64),
            size=surrogate.remaining,
            replace=False,
        )
        completion_scores, completion_stds = surrogate.query(completion_codes)
        archive.add(
            completion_codes,
            completion_scores,
            completion_stds,
            step=round_index + 1,
        )
    return _finalize_black_box(
        "adalead",
        archive,
        surrogate,
        visible_set,
        config,
        threshold=threshold,
        rounds=round_index,
        expected_mutations=1,
        full_space_novelty_reserve=novelty_reserve,
    )


def _one_hot_for_codes(codes: np.ndarray, length: int, device: torch.device) -> torch.Tensor:
    encoded = codes_to_encoded(np.asarray(codes, dtype=np.int64), length)
    return F.one_hot(
        torch.as_tensor(encoded, dtype=torch.long, device=device), num_classes=4
    ).float()


def _codes_for_one_hot(one_hot: torch.Tensor) -> np.ndarray:
    encoded = one_hot.detach().argmax(dim=-1).cpu().numpy().astype(np.uint8)
    return encoded_to_codes(encoded)


def _straight_through_sample(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = probabilities.shape
    sampled = torch.multinomial(probabilities.reshape(-1, 4), 1).reshape(shape[:-1])
    hard = F.one_hot(sampled, num_classes=4).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach(), hard


def _rank_proxy_pool(
    proxy_records: dict[int, float],
    visible_set: set[int],
    count: int,
) -> np.ndarray:
    eligible = [(code, value) for code, value in proxy_records.items() if code not in visible_set]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if len(eligible) < count:
        raise RuntimeError(f"Activation maximization produced only {len(eligible)} novel discrete designs")
    return np.asarray([item[0] for item in eligible[:count]], dtype=np.int64)


def optimize_vanilla_activation_maximization(
    ensemble_model: DifferentiableTFBindEnsemble,
    cached_scores: np.ndarray,
    cached_stds: np.ndarray,
    ranked_codes: np.ndarray,
    visible_count: int,
    length: int,
    config: TFBind8OptimizationConfig,
    seed: int,
    initial_codes: np.ndarray | None = None,
) -> OptimizationResult:
    rng = _set_seed(seed)
    device = ensemble_model.label_means.device
    visible_set = set(map(int, ranked_codes[:visible_count]))
    initial = _resolve_initial_codes(
        initial_codes, ranked_codes, visible_count, len(cached_scores), config, rng
    )
    initial_one_hot = _one_hot_for_codes(initial, length, device)
    logits = nn.Parameter(initial_one_hot * 1.5 + 0.05 * torch.randn_like(initial_one_hot))
    optimizer = torch.optim.Adam([logits], lr=config.gradient_learning_rate)
    gradient_steps = (config.query_budget - config.num_candidates) // config.num_candidates
    proxy_records: dict[int, float] = {}
    for _ in range(gradient_steps):
        optimizer.zero_grad(set_to_none=True)
        probabilities = torch.softmax(logits, dim=-1)
        prediction, _ = ensemble_model(probabilities)
        (-prediction.mean()).backward()
        optimizer.step()
        codes = _codes_for_one_hot(probabilities)
        for code, value in zip(codes, prediction.detach().cpu().numpy()):
            proxy_records[int(code)] = max(float(value), proxy_records.get(int(code), -np.inf))
    # The original PWM activation-maximization method interprets the optimized
    # softmax matrices as distributions and draws discrete sequence designs.
    # Add those samples to the proxy pool before spending the final 100 calls
    # on discrete candidates.
    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=-1)
        final_proxy = ensemble_model(probabilities)[0].detach().cpu().numpy()
        attempts = 0
        while len([code for code in proxy_records if code not in visible_set]) < config.num_candidates:
            sampled = torch.multinomial(probabilities.reshape(-1, 4), 1).reshape(
                config.num_candidates, length
            )
            sampled_codes = encoded_to_codes(sampled.cpu().numpy().astype(np.uint8))
            for code, value in zip(sampled_codes, final_proxy):
                proxy_records[int(code)] = max(float(value), proxy_records.get(int(code), -np.inf))
            attempts += 1
            if attempts >= 1000:
                break
    novel_proxy_count = len([code for code in proxy_records if code not in visible_set])
    if novel_proxy_count < config.num_candidates:
        # This is only reachable after extreme PWM collapse. Fill the proposal
        # pool without consulting the surrogate; the final discrete calls below
        # still determine the weak scores under the common budget.
        available = np.asarray(
            [code for code in range(len(cached_scores)) if code not in visible_set and code not in proxy_records],
            dtype=np.int64,
        )
        fill = rng.choice(available, size=config.num_candidates - novel_proxy_count, replace=False)
        for code in fill:
            proxy_records[int(code)] = -np.inf
    final_codes = _rank_proxy_pool(proxy_records, visible_set, config.num_candidates)
    final_scores = cached_scores[final_codes]
    final_stds = cached_stds[final_codes]
    query_count = gradient_steps * config.num_candidates + len(final_codes)
    return OptimizationResult(
        "vanilla_activation_maximization",
        final_codes,
        final_scores,
        final_stds,
        query_count,
        {
            "gradient_steps": gradient_steps,
            "learning_rate": config.gradient_learning_rate,
            "final_pwm_sampling_attempts": attempts,
            "uniform_fallback_count": max(0, config.num_candidates - novel_proxy_count),
        },
    )


def _fast_seqprop_logits(logits: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # DNA Fast SeqProp uses instance normalization: each nucleotide channel is
    # normalized independently across sequence positions (Linder & Seelig 2021).
    mean = logits.mean(dim=1, keepdim=True)
    variance = logits.var(dim=1, keepdim=True, unbiased=False)
    normalized = (logits - mean) / torch.sqrt(variance + 1e-5)
    return normalized * gamma[:, None, :] + beta[:, None, :]


def optimize_fast_seqprop(
    ensemble_model: DifferentiableTFBindEnsemble,
    ranked_codes: np.ndarray,
    visible_count: int,
    landscape_size: int,
    length: int,
    config: TFBind8OptimizationConfig,
    seed: int,
    initial_codes: np.ndarray | None = None,
) -> OptimizationResult:
    rng = _set_seed(seed)
    device = ensemble_model.label_means.device
    visible_set = set(map(int, ranked_codes[:visible_count]))
    initial = _resolve_initial_codes(
        initial_codes, ranked_codes, visible_count, landscape_size, config, rng
    )
    initial_one_hot = _one_hot_for_codes(initial, length, device)
    logits = nn.Parameter(initial_one_hot + 0.05 * torch.randn_like(initial_one_hot))
    gamma = nn.Parameter(torch.ones(config.num_candidates, 4, device=device))
    beta = nn.Parameter(torch.zeros(config.num_candidates, 4, device=device))
    optimizer = torch.optim.Adam([logits, gamma, beta], lr=config.gradient_learning_rate)
    archive = CandidateArchive()
    steps = config.query_budget // config.num_candidates
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        scaled = _fast_seqprop_logits(logits, gamma, beta)
        probabilities = torch.softmax(scaled, dim=-1)
        straight_through, hard = _straight_through_sample(probabilities)
        prediction, uncertainty = ensemble_model(straight_through)
        (-prediction.mean()).backward()
        optimizer.step()
        archive.add(
            _codes_for_one_hot(hard),
            prediction.detach().cpu().numpy(),
            uncertainty.detach().cpu().numpy(),
            step=step,
        )
    codes, scores, stds = archive.ranked(visible_set, config.num_candidates)
    return OptimizationResult(
        "fast_seqprop",
        codes,
        scores,
        stds,
        steps * config.num_candidates,
        {"gradient_steps": steps, "learning_rate": config.gradient_learning_rate},
    )


class AutoregressiveDNA(nn.Module):
    """Small exact-likelihood DNA generator used as the CbAS search density."""

    def __init__(self, length: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.length = length
        self.embedding = nn.Embedding(5, 16)
        self.gru = nn.GRU(16, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 4)

    def logits(self, encoded: torch.Tensor) -> torch.Tensor:
        start = torch.full((len(encoded), 1), 4, dtype=torch.long, device=encoded.device)
        inputs = torch.cat([start, encoded[:, :-1]], dim=1)
        hidden, _ = self.gru(self.embedding(inputs))
        return self.output(hidden)

    def log_prob(self, encoded: torch.Tensor) -> torch.Tensor:
        logits = self.logits(encoded)
        return -F.cross_entropy(logits.transpose(1, 2), encoded, reduction="none").sum(dim=1)

    @torch.no_grad()
    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        token = torch.full((count, 1), 4, dtype=torch.long, device=device)
        hidden = None
        output = []
        for _ in range(self.length):
            state, hidden = self.gru(self.embedding(token), hidden)
            logits = self.output(state[:, -1])
            base = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            output.append(base)
            token = base
        return torch.cat(output, dim=1)


def _fit_autoregressive(
    model: AutoregressiveDNA,
    encoded: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    weights: torch.Tensor | None = None,
    batch_size: int = 1024,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    count = len(encoded)
    for _ in range(epochs):
        permutation = torch.randperm(count, device=encoded.device)
        for start in range(0, count, batch_size):
            indices = permutation[start : start + batch_size]
            log_probability = model.log_prob(encoded[indices])
            if weights is None:
                loss = -log_probability.mean()
            else:
                batch_weights = weights[indices]
                loss = -(batch_weights * log_probability).sum() / batch_weights.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


def optimize_cbas(
    cached_scores: np.ndarray,
    cached_stds: np.ndarray,
    ranked_codes: np.ndarray,
    visible_count: int,
    length: int,
    config: TFBind8OptimizationConfig,
    seed: int,
    device: torch.device,
    initial_codes: np.ndarray | None = None,
) -> OptimizationResult:
    rng = _set_seed(seed)
    visible_set = set(map(int, ranked_codes[:visible_count]))
    initial = _resolve_initial_codes(
        initial_codes, ranked_codes, visible_count, len(cached_scores), config, rng
    )
    initial_encoded = torch.as_tensor(
        codes_to_encoded(initial, length).astype(np.int64), device=device
    )
    prior = AutoregressiveDNA(length).to(device)
    _fit_autoregressive(
        prior,
        initial_encoded,
        epochs=config.cbas_pretrain_epochs,
        learning_rate=3e-3,
    )
    search = copy.deepcopy(prior)
    archive = CandidateArchive()
    surrogate = BudgetedLookupSurrogate(cached_scores, cached_stds, config.query_budget)
    batch_size = config.query_budget // config.cbas_rounds
    if batch_size < config.num_candidates:
        raise ValueError("CbAS requires at least num_candidates queries per round")
    threshold = -np.inf
    completed_rounds = 0
    while surrogate.remaining > 0:
        current_batch = min(batch_size, surrogate.remaining)
        samples = search.sample(current_batch, device)
        codes = encoded_to_codes(samples.cpu().numpy().astype(np.uint8))
        prediction, uncertainty = surrogate.query(codes)
        completed_rounds += 1
        archive.add(codes, prediction, uncertainty, step=completed_rounds)
        threshold = max(threshold, float(np.quantile(prediction, config.cbas_quantile)))
        standard_deviation = np.maximum(uncertainty, max(float(np.median(uncertainty)) * 0.1, 1e-3))
        success_probability = np.clip(ndtr((prediction - threshold) / standard_deviation), 1e-12, 1.0)
        with torch.no_grad():
            log_prior = prior.log_prob(samples).cpu().numpy()
            log_search = search.log_prob(samples).cpu().numpy()
        log_weights = np.log(success_probability) + log_prior - log_search
        log_weights = np.clip(log_weights - log_weights.max(), -30, 0)
        weights = torch.as_tensor(np.exp(log_weights), dtype=torch.float32, device=device)
        _fit_autoregressive(
            search,
            samples,
            epochs=config.cbas_update_epochs,
            learning_rate=1e-3,
            weights=weights,
            batch_size=min(1024, current_batch),
        )
    codes, scores, stds = archive.ranked(visible_set, config.num_candidates)
    return OptimizationResult(
        "cbas",
        codes,
        scores,
        stds,
        surrogate.query_count,
        {
            "rounds": completed_rounds,
            "batch_size": batch_size,
            "conditioning_quantile": config.cbas_quantile,
            "final_threshold": threshold,
            "generator": "exact-likelihood autoregressive GRU",
            "initial_generator_fit_sequences": len(initial),
        },
    )


class DeepExplorationGenerator(nn.Module):
    """Noise-to-categorical-logit generator for DEN optimization."""

    def __init__(self, noise_dim: int, length: int) -> None:
        super().__init__()
        self.length = length
        self.network = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, length * 4),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise).reshape(len(noise), self.length, 4)


def optimize_den(
    ensemble_model: DifferentiableTFBindEnsemble,
    ranked_codes: np.ndarray,
    visible_count: int,
    length: int,
    config: TFBind8OptimizationConfig,
    seed: int,
    initial_codes: np.ndarray | None = None,
) -> OptimizationResult:
    rng = _set_seed(seed)
    device = ensemble_model.label_means.device
    visible_set = set(map(int, ranked_codes[:visible_count]))
    initial = _resolve_initial_codes(
        initial_codes,
        ranked_codes,
        visible_count,
        4**length,
        config,
        rng,
    )
    noise_dim = 16
    batch_size = config.num_candidates
    generator = DeepExplorationGenerator(noise_dim, length).to(device)
    fixed_noise = torch.randn(batch_size, noise_dim, device=device)
    target = torch.as_tensor(
        codes_to_encoded(initial, length).astype(np.int64),
        dtype=torch.long,
        device=device,
    )
    seed_optimizer = torch.optim.Adam(generator.parameters(), lr=3e-3)
    for _ in range(config.den_seed_pretrain_epochs):
        seed_logits = generator(fixed_noise)
        seed_loss = F.cross_entropy(seed_logits.reshape(-1, 4), target.reshape(-1))
        seed_optimizer.zero_grad(set_to_none=True)
        seed_loss.backward()
        seed_optimizer.step()
    with torch.no_grad():
        reconstructed = generator(fixed_noise).argmax(dim=-1)
        reconstruction_accuracy = float((reconstructed == target).float().mean().cpu())

    optimizer = torch.optim.Adam(generator.parameters(), lr=3e-3)
    archive = CandidateArchive()
    steps = config.query_budget // batch_size
    score_scale = max(float(ensemble_model.label_stds.mean().detach().cpu()), 1e-3)
    for step in range(steps):
        logits = generator(fixed_noise)
        probabilities = torch.softmax(logits, dim=-1)
        straight_through, hard = _straight_through_sample(probabilities)
        prediction, uncertainty = ensemble_model(straight_through)
        half = batch_size // 2
        if half:
            pair_similarity = (straight_through[:half] * straight_through[half : 2 * half]).sum(dim=-1).mean()
        else:
            pair_similarity = torch.zeros((), device=device)
        loss = -prediction.mean() / score_scale + config.den_diversity_weight * pair_similarity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        optimizer.step()
        archive.add(
            _codes_for_one_hot(hard),
            prediction.detach().cpu().numpy(),
            uncertainty.detach().cpu().numpy(),
            step=step,
        )
    codes, scores, stds = archive.ranked(visible_set, config.num_candidates)
    return OptimizationResult(
        "den",
        codes,
        scores,
        stds,
        steps * batch_size,
        {
            "steps": steps,
            "batch_size": batch_size,
            "noise_dim": noise_dim,
            "diversity_weight": config.den_diversity_weight,
            "seed_pretrain_epochs": config.den_seed_pretrain_epochs,
            "seed_reconstruction_accuracy": reconstruction_accuracy,
        },
    )


BLACK_BOX_OPTIMIZERS: dict[str, Callable[..., OptimizationResult]] = {
    "random_screening": optimize_random_screening,
    "ism_greedy": optimize_ism_greedy,
    "simulated_annealing": optimize_simulated_annealing,
    "genetic_algorithm": optimize_genetic_algorithm,
    "adalead": optimize_adalead,
}


def _pairwise_hamming(encoded: np.ndarray) -> float:
    if len(encoded) < 2:
        return 0.0
    differences = encoded[:, None, :] != encoded[None, :, :]
    upper = np.triu_indices(len(encoded), k=1)
    return float(differences.sum(axis=-1)[upper].mean())


def _minimum_training_hamming(candidate_encoded: np.ndarray, training_encoded: np.ndarray) -> np.ndarray:
    minima = np.full(len(candidate_encoded), candidate_encoded.shape[1], dtype=np.int64)
    chunk_size = 4096
    for start in range(0, len(training_encoded), chunk_size):
        chunk = training_encoded[start : start + chunk_size]
        distances = (candidate_encoded[:, None, :] != chunk[None, :, :]).sum(axis=-1)
        minima = np.minimum(minima, distances.min(axis=1))
    return minima


def _evaluate_result(
    result: OptimizationResult,
    landscape: TFBindLandscape,
    train_fraction: float,
    visible_codes: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int | bool | str | None]]:
    sequences = landscape.sequences_for_codes(result.codes).tolist()
    scored, exact_summary = evaluate_candidates(
        "tfbind8",
        sequences,
        train_fraction=train_fraction,
        top_k=100,
    )
    scored["weak_prediction"] = result.weak_scores
    scored["weak_uncertainty"] = result.weak_stds
    scored["prediction_error"] = scored["weak_prediction"] - scored["true_score"]
    scored["algorithm"] = result.algorithm
    scored["weak_model_fraction"] = train_fraction
    scored["query_count"] = result.query_count
    encoded = codes_to_encoded(result.codes, landscape.length)
    training_encoded = codes_to_encoded(visible_codes, landscape.length)
    min_training_hamming = _minimum_training_hamming(encoded, training_encoded)
    scored["minimum_training_hamming"] = min_training_hamming
    true_scores = scored["true_score"].to_numpy()
    ordered = np.sort(true_scores)[::-1]
    activity_percentiles = scored["activity_percentile"].to_numpy()
    summary = {
        **exact_summary,
        "algorithm": result.algorithm,
        "weak_model_fraction": train_fraction,
        "query_count": result.query_count,
        "mean_weak_prediction": float(np.mean(result.weak_scores)),
        "max_weak_prediction": float(np.max(result.weak_scores)),
        "mean_prediction_error": float(np.mean(result.weak_scores - true_scores)),
        "mean_absolute_prediction_error": float(np.mean(np.abs(result.weak_scores - true_scores))),
        "median_true_score": float(np.median(true_scores)),
        "top_10_mean_true_score": float(ordered[:10].mean()),
        "mean_activity_percentile": float(activity_percentiles.mean()),
        "top_5pct_count": int(np.sum(activity_percentiles >= 95.0)),
        "top_1pct_count": int(np.sum(activity_percentiles >= 99.0)),
        "top_0_1pct_count": int(np.sum(activity_percentiles >= 99.9)),
        "mean_pairwise_hamming": _pairwise_hamming(encoded),
        "mean_minimum_training_hamming": float(min_training_hamming.mean()),
        "optimizer_metadata": json.dumps(result.metadata, sort_keys=True),
    }
    return scored, summary


def _load_or_compute_predictions(
    ensemble: TFBindWeakEnsemble,
    landscape: TFBindLandscape,
    cache_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            mean = np.asarray(archive["mean"], dtype=np.float64)
            std = np.asarray(archive["std"], dtype=np.float64)
        if mean.shape == (landscape.size,) and std.shape == (landscape.size,):
            return mean, std
    sequences = landscape.sequences_for_codes(np.arange(landscape.size)).tolist()
    mean, std = ensemble.predict(sequences, batch_size=batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, mean=mean, std=std)
    return mean.astype(np.float64), std.astype(np.float64)


def run_tfbind8_benchmark(
    *,
    checkpoint_root: str | Path = "BioSeqBench/results/tfbind_weak_cnns/tfbind8",
    output_dir: str | Path = "BioSeqBench/results/tfbind8_optimization_benchmark",
    raw_path: str | Path | None = None,
    device: str | torch.device | None = None,
    config: TFBind8OptimizationConfig | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Run all requested weak-model/optimizer combinations and write results."""
    config = config or TFBind8OptimizationConfig()
    config.validate()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_root)
    compute_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    landscape = load_tfbind_landscape("tfbind8", raw_path)
    ranked_codes = landscape.ranked_codes()
    protocol = {
        **asdict(config),
        "algorithms": list(config.algorithms),
        "train_fractions": list(config.train_fractions),
        "selected_weak_models": [_fraction_name(value) for value in config.train_fractions],
        "candidate_policy": (
            f"optimizers return exactly {config.num_candidates} unique sequences outside the visible "
            f"lower-tail pool; training_top100 returns the {config.num_candidates} highest-activity "
            "sequences inside that pool as an observed-data reference"
        ),
        "initialization_policy": (
            f"all optimizers except random_screening share the same {config.num_candidates} seeds "
            f"within each model/replicate: {config.top_visible_initializations} highest-activity "
            f"training sequences plus {config.random_initializations} uniform full-space random "
            "sequences sampled without conditioning on training membership; training_top100 is an "
            "observed-data reference rather than an optimizer"
        ),
        "evaluation_policy": "optimization uses only the fixed CNN weak ensemble; exact landscape queried only after generation",
        "device": str(compute_device),
    }
    (output_path / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries: list[dict[str, float | int | bool | str | None]] = []
    all_candidates: list[pd.DataFrame] = []
    all_initializations: list[pd.DataFrame] = []

    for fraction_index, fraction in enumerate(config.train_fractions):
        model_name = _fraction_name(fraction)
        model_output = output_path / model_name
        model_output.mkdir(parents=True, exist_ok=True)
        print(f"Loading {model_name} on {compute_device}", flush=True)
        ensemble = TFBindWeakEnsemble(checkpoint_path / model_name, device=compute_device)
        prediction_cache = model_output / "all_landscape_weak_predictions.npz"
        cached_scores, cached_stds = _load_or_compute_predictions(
            ensemble,
            landscape,
            prediction_cache,
            config.prediction_batch_size,
        )
        differentiable = DifferentiableTFBindEnsemble(ensemble).to(compute_device)
        visible_count = int(round(landscape.size * fraction))
        visible_codes = ranked_codes[:visible_count]

        for replicate in range(config.num_replicates):
            replicate_output = model_output / f"run_{replicate:02d}"
            replicate_output.mkdir(parents=True, exist_ok=True)
            initialization_seed = (
                config.base_seed + fraction_index * 100_000 + replicate * 1_000 + 997
            )
            shared_initial_codes, initial_sources = _shared_initial_codes(
                ranked_codes,
                visible_count,
                landscape.size,
                config,
                np.random.default_rng(initialization_seed),
            )
            initialization_id = hashlib.sha256(
                np.asarray(shared_initial_codes, dtype=np.int64).tobytes()
            ).hexdigest()[:16]
            initial_frame = pd.DataFrame(
                {
                    "sequence_code": shared_initial_codes,
                    "sequence": landscape.sequences_for_codes(shared_initial_codes),
                    "source": initial_sources,
                    "in_weak_train": np.isin(shared_initial_codes, visible_codes),
                    "initialization_seed": initialization_seed,
                    "initialization_id": initialization_id,
                }
            )
            initial_frame["weak_model_fraction"] = fraction
            initial_frame["replicate"] = replicate
            initial_frame.to_csv(replicate_output / "shared_initial_sequences.csv", index=False)
            all_initializations.append(initial_frame)
            for algorithm_index, algorithm in enumerate(config.algorithms):
                algorithm_seed = (
                    config.base_seed
                    + fraction_index * 100_000
                    + replicate * 1_000
                    + algorithm_index * 101
                )
                candidate_path = replicate_output / f"{algorithm}_candidates.csv"
                summary_path = replicate_output / f"{algorithm}_summary.json"
                if resume and candidate_path.exists() and summary_path.exists():
                    scored = pd.read_csv(candidate_path)
                    summary = json.loads(summary_path.read_text())
                    summaries.append(summary)
                    all_candidates.append(scored)
                    print(
                        f"Skipping completed {model_name} run={replicate:02d} {algorithm}",
                        flush=True,
                    )
                    continue

                print(
                    f"Optimizing {model_name} run={replicate:02d} with {algorithm}",
                    flush=True,
                )
                if algorithm == "training_top100":
                    result = optimize_training_top100(
                        cached_scores,
                        cached_stds,
                        ranked_codes,
                        visible_count,
                        config,
                    )
                elif algorithm in BLACK_BOX_OPTIMIZERS:
                    optimizer_initial = (
                        None if algorithm == "random_screening" else shared_initial_codes
                    )
                    result = BLACK_BOX_OPTIMIZERS[algorithm](
                        cached_scores,
                        cached_stds,
                        ranked_codes,
                        visible_count,
                        landscape.length,
                        config,
                        algorithm_seed,
                        optimizer_initial,
                    )
                elif algorithm == "vanilla_activation_maximization":
                    result = optimize_vanilla_activation_maximization(
                        differentiable,
                        cached_scores,
                        cached_stds,
                        ranked_codes,
                        visible_count,
                        landscape.length,
                        config,
                        algorithm_seed,
                        shared_initial_codes,
                    )
                elif algorithm == "fast_seqprop":
                    result = optimize_fast_seqprop(
                        differentiable,
                        ranked_codes,
                        visible_count,
                        landscape.size,
                        landscape.length,
                        config,
                        algorithm_seed,
                        shared_initial_codes,
                    )
                elif algorithm == "cbas":
                    result = optimize_cbas(
                        cached_scores,
                        cached_stds,
                        ranked_codes,
                        visible_count,
                        landscape.length,
                        config,
                        algorithm_seed,
                        compute_device,
                        shared_initial_codes,
                    )
                elif algorithm == "den":
                    result = optimize_den(
                        differentiable,
                        ranked_codes,
                        visible_count,
                        landscape.length,
                        config,
                        algorithm_seed,
                        shared_initial_codes,
                    )
                else:  # pragma: no cover - guarded by config validation.
                    raise AssertionError(algorithm)
                expected_queries = 0 if algorithm == "training_top100" else config.query_budget
                if result.query_count != expected_queries:
                    raise RuntimeError(
                        f"{algorithm} used {result.query_count} weak-model queries; "
                        f"the expected count is {expected_queries}"
                    )
                if len(result.codes) != config.num_candidates or len(np.unique(result.codes)) != config.num_candidates:
                    raise RuntimeError(f"{algorithm} did not return exactly {config.num_candidates} unique candidates")
                in_training_count = np.intersect1d(result.codes, visible_codes).size
                if algorithm == "training_top100" and in_training_count != config.num_candidates:
                    raise RuntimeError("training_top100 returned a sequence outside the visible training pool")
                if algorithm != "training_top100" and in_training_count:
                    raise RuntimeError(f"{algorithm} returned a sequence from the visible weak training pool")
                scored, summary = _evaluate_result(result, landscape, fraction, visible_codes)
                scored["replicate"] = replicate
                scored["optimizer_seed"] = algorithm_seed
                scored["initialization_seed"] = initialization_seed
                scored["initialization_id"] = initialization_id
                scored["uses_shared_initialization"] = algorithm not in {
                    "training_top100",
                    "random_screening",
                }
                summary["replicate"] = replicate
                summary["optimizer_seed"] = algorithm_seed
                summary["initialization_seed"] = initialization_seed
                summary["initialization_id"] = initialization_id
                summary["uses_shared_initialization"] = algorithm not in {
                    "training_top100",
                    "random_screening",
                }
                scored.to_csv(candidate_path, index=False)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                summaries.append(summary)
                all_candidates.append(scored)
                print(
                    f"  true max={summary['best_true_score']:.5f}, "
                    f"mean={summary['mean_true_score']:.5f}, top1%={summary['top_1pct_count']}",
                    flush=True,
                )

        del differentiable, ensemble
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_path / "summary.csv", index=False)
    candidate_frame = pd.concat(all_candidates, ignore_index=True)
    candidate_frame.to_csv(output_path / "all_candidates.csv", index=False)
    initialization_frame = pd.concat(all_initializations, ignore_index=True)
    initialization_frame.to_csv(output_path / "all_shared_initial_sequences.csv", index=False)
    (
        initialization_frame.groupby(
            ["weak_model_fraction", "replicate", "source"], sort=False
        )
        .agg(
            sequence_count=("sequence", "size"),
            in_weak_train_count=("in_weak_train", "sum"),
            in_weak_train_fraction=("in_weak_train", "mean"),
            initialization_id=("initialization_id", "first"),
        )
        .reset_index()
        .to_csv(output_path / "shared_initialization_summary.csv", index=False)
    )
    _aggregate_replicate_results(summary_frame).to_csv(
        output_path / "aggregate_summary.csv", index=False
    )
    return summary_frame


def _aggregate_replicate_results(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate optimizer metrics across independent replicate seeds."""
    random_means = summary.loc[
        summary["algorithm"] == "random_screening",
        ["weak_model_fraction", "replicate", "mean_true_score"],
    ].rename(columns={"mean_true_score": "random_mean_true_score"})
    enriched = summary.merge(
        random_means,
        on=["weak_model_fraction", "replicate"],
        how="left",
        validate="many_to_one",
    )
    enriched["mean_activity_gain_over_random"] = (
        enriched["mean_true_score"] - enriched["random_mean_true_score"]
    )
    metrics = (
        "mean_true_score",
        "median_true_score",
        "top_10_mean_true_score",
        "best_true_score",
        "top_5pct_count",
        "top_1pct_count",
        "top_0_1pct_count",
        "mean_pairwise_hamming",
        "mean_minimum_training_hamming",
        "mean_absolute_prediction_error",
        "mean_activity_gain_over_random",
    )
    named_aggregations: dict[str, tuple[str, str]] = {
        "replicates": ("replicate", "nunique")
    }
    for metric in metrics:
        named_aggregations[f"{metric}_mean"] = (metric, "mean")
        named_aggregations[f"{metric}_std"] = (metric, "std")
        named_aggregations[f"{metric}_sem"] = (metric, "sem")
    return (
        enriched.groupby(["dataset", "weak_model_fraction", "algorithm"], sort=False)
        .agg(**named_aggregations)
        .reset_index()
    )
