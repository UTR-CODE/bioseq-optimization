"""Model-agnostic sequence optimization interfaces.

The public contract is intentionally small:

* a :class:`SequenceSpace` validates/encodes a discrete alphabet;
* a :class:`SequenceScorer` maps ``list[str]`` to scores;
* an optimizer maps initial sequences plus a scorer to an
  :class:`OptimizationResult`.

Models never need to know which optimizer is being used.  Conversely, an
optimizer never imports a dataset-specific model or oracle.  New algorithms can
be added with ``@register_optimizer("name")``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class SequenceScorer(Protocol):
    """Minimal black-box scoring interface used by every optimizer."""

    def score(self, sequences: Sequence[str] | np.ndarray) -> "ScoreBatch":
        """Return one score (and optionally uncertainty) per sequence."""


@dataclass(frozen=True)
class ScoreBatch:
    scores: np.ndarray
    uncertainties: np.ndarray | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError("scorer returned non-finite scores")
        uncertainties = self.uncertainties
        if uncertainties is not None:
            uncertainties = np.asarray(uncertainties, dtype=np.float64).reshape(-1)
            if len(uncertainties) != len(scores):
                raise ValueError("uncertainties must have one value per score")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "uncertainties", uncertainties)


class CallableScorer:
    """Adapt any callable returning scores or ``(scores, uncertainties)``."""

    def __init__(self, function: Callable[[Sequence[str] | np.ndarray], Any]) -> None:
        self.function = function

    def score(self, sequences: Sequence[str] | np.ndarray) -> ScoreBatch:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        result = self.function(values)
        if isinstance(result, ScoreBatch):
            return result
        if isinstance(result, tuple) and len(result) == 2:
            return ScoreBatch(result[0], result[1])
        return ScoreBatch(result)


class TorchSequenceScorer:
    """Wrap a frozen PyTorch sequence model behind :class:`SequenceScorer`.

    ``encoder`` receives a NumPy string array and returns the model input tensor
    (on CPU or directly on ``device``). ``predictor`` may return a tensor, a
    ``(scores, uncertainties)`` tuple, or a ``ScoreBatch``. The adapter keeps
    gradients disabled because black-box optimizers consume only scores; users
    implementing gradient optimizers can expose their own differentiable model
    wrapper alongside this scorer.
    """

    def __init__(
        self,
        model: Any,
        encoder: Callable[[np.ndarray], Any],
        *,
        device: Any = None,
        batch_size: int = 4096,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        if hasattr(model, "eval"):
            model.eval()

    def score(self, sequences: Sequence[str] | np.ndarray) -> ScoreBatch:
        import torch

        values = np.asarray(sequences, dtype=str).reshape(-1)
        outputs: list[np.ndarray] = []
        uncertainties: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                inputs = self.encoder(batch)
                if self.device is not None and hasattr(inputs, "to"):
                    inputs = inputs.to(self.device)
                result = self.model(inputs)
                if isinstance(result, ScoreBatch):
                    score_batch = result
                elif isinstance(result, tuple) and len(result) == 2:
                    score_batch = ScoreBatch(
                        result[0].detach().cpu().numpy() if hasattr(result[0], "detach") else result[0],
                        result[1].detach().cpu().numpy() if hasattr(result[1], "detach") else result[1],
                    )
                else:
                    scores = result.detach().cpu().numpy() if hasattr(result, "detach") else result
                    score_batch = ScoreBatch(scores)
                outputs.append(score_batch.scores)
                if score_batch.uncertainties is not None:
                    uncertainties.append(score_batch.uncertainties)
        return ScoreBatch(
            np.concatenate(outputs) if outputs else np.empty(0),
            np.concatenate(uncertainties) if uncertainties else np.zeros(len(values)),
        )


class CachedScorer:
    """Cache model calls and enforce a query budget.

    By default the budget counts unique sequences. Repeated requests are served
    from the cache and do not consume additional model evaluations. Set
    ``count_repeated=True`` when an external service charges every request.
    """

    def __init__(
        self,
        scorer: SequenceScorer,
        budget: int | None = None,
        *,
        count_repeated: bool = False,
    ) -> None:
        if budget is not None and budget < 0:
            raise ValueError("budget must be non-negative or None")
        self.scorer = scorer
        self.budget = budget
        self.count_repeated = count_repeated
        self._cache: dict[str, tuple[float, float]] = {}
        self.query_count = 0

    @property
    def remaining(self) -> int | None:
        return None if self.budget is None else self.budget - self.query_count

    def score(self, sequences: Sequence[str] | np.ndarray) -> ScoreBatch:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        if not len(values):
            return ScoreBatch(np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
        missing: list[str] = []
        missing_set: set[str] = set()
        for value in values:
            key = str(value)
            if key not in self._cache and key not in missing_set:
                missing.append(key)
                missing_set.add(key)
        cost = len(values) if self.count_repeated else len(missing)
        if self.budget is not None and cost > self.remaining:
            raise BudgetExceeded(
                f"query budget exceeded: requested {cost}, remaining {self.remaining}"
            )
        if missing:
            result = self.scorer.score(np.asarray(missing, dtype=str))
            if len(result.scores) != len(missing):
                raise ValueError("scorer returned the wrong number of scores")
            uncertainties = (
                np.zeros(len(missing), dtype=np.float64)
                if result.uncertainties is None
                else result.uncertainties
            )
            self._cache.update(
                {key: (float(score), float(std)) for key, score, std in zip(missing, result.scores, uncertainties)}
            )
        self.query_count += cost
        scores = np.asarray([self._cache[str(value)][0] for value in values], dtype=np.float64)
        uncertainties = np.asarray([self._cache[str(value)][1] for value in values], dtype=np.float64)
        return ScoreBatch(scores, uncertainties)

    def cached(self, sequences: Sequence[str] | np.ndarray) -> bool:
        return all(str(value) in self._cache for value in np.asarray(sequences, dtype=str).reshape(-1))


class BudgetExceeded(RuntimeError):
    """Raised when an optimizer asks for more model evaluations than allowed."""


@dataclass(frozen=True)
class SequenceSpace:
    length: int
    alphabet: str = "ACGT"

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError("sequence length must be positive")
        if len(self.alphabet) < 2 or len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet must contain at least two unique symbols")

    @property
    def alphabet_size(self) -> int:
        return len(self.alphabet)

    def validate(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        if np.any(np.char.str_len(values) != self.length):
            raise ValueError(f"all sequences must have length {self.length}")
        allowed = set(self.alphabet)
        for value in values:
            if set(value) - allowed:
                raise ValueError(f"sequence contains symbols outside alphabet {self.alphabet!r}: {value!r}")
        return values

    def encode(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        values = self.validate(sequences)
        lookup = {symbol: index for index, symbol in enumerate(self.alphabet)}
        return np.asarray([[lookup[symbol] for symbol in value] for value in values], dtype=np.int16)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        values = np.asarray(encoded, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != self.length:
            raise ValueError(f"encoded sequences must have shape [N, {self.length}]")
        if np.any(values < 0) or np.any(values >= self.alphabet_size):
            raise ValueError("encoded sequence contains an invalid alphabet index")
        symbols = np.asarray(list(self.alphabet))
        return np.asarray(["".join(symbols[row]) for row in values], dtype=str)

    def random(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        return self.decode(rng.integers(0, self.alphabet_size, size=(count, self.length)))

    def single_mutants(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        encoded = self.encode(sequences)
        blocks = []
        for sequence in encoded:
            block = []
            for position in range(self.length):
                for value in range(self.alphabet_size):
                    if value != sequence[position]:
                        candidate = sequence.copy()
                        candidate[position] = value
                        block.append(candidate)
            blocks.append(np.asarray(block, dtype=np.int16))
        mutants = np.asarray(blocks, dtype=np.int16)
        return self.decode(mutants.reshape(-1, self.length))

    def mutate_one(self, sequences: Sequence[str] | np.ndarray, rng: np.random.Generator) -> np.ndarray:
        encoded = self.encode(sequences).copy()
        rows = np.arange(len(encoded))
        positions = rng.integers(0, self.length, len(encoded))
        offsets = rng.integers(1, self.alphabet_size, len(encoded))
        encoded[rows, positions] = (encoded[rows, positions] + offsets) % self.alphabet_size
        return self.decode(encoded)


@dataclass(frozen=True)
class OptimizationConfig:
    budget: int
    num_candidates: int = 100
    population_size: int | None = None
    mutation_rate: float = 0.02
    elite_fraction: float = 0.2
    learning_rate: float = 0.1
    initial_temperature: float | None = None
    min_temperature: float = 1e-6

    def __post_init__(self) -> None:
        if self.budget < 1 or self.num_candidates < 1 or self.budget < self.num_candidates:
            raise ValueError("budget must be at least num_candidates and both must be positive")
        population = self.population_size or self.num_candidates
        if population < 2:
            raise ValueError("population_size must be at least two")
        if not 0 < self.mutation_rate <= 1 or not 0 < self.elite_fraction <= 1:
            raise ValueError("mutation_rate and elite_fraction must lie in (0, 1]")
        if self.learning_rate <= 0 or self.min_temperature <= 0:
            raise ValueError("learning_rate and min_temperature must be positive")


@dataclass
class OptimizationResult:
    algorithm: str
    sequences: np.ndarray
    scores: np.ndarray
    uncertainties: np.ndarray
    query_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, float | int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sequences = np.asarray(self.sequences, dtype=str).reshape(-1)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        self.uncertainties = np.asarray(self.uncertainties, dtype=np.float64).reshape(-1)
        if not (len(self.sequences) == len(self.scores) == len(self.uncertainties)):
            raise ValueError("sequences, scores, and uncertainties must have equal lengths")
        if len(np.unique(self.sequences)) != len(self.sequences):
            raise ValueError("optimization result contains duplicate sequences")

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame({"sequence": self.sequences, "score": self.scores, "uncertainty": self.uncertainties})


Optimizer = Callable[[np.ndarray, CachedScorer, SequenceSpace, OptimizationConfig, np.random.Generator], OptimizationResult]
_REGISTRY: dict[str, Optimizer] = {}


def register_optimizer(name: str) -> Callable[[Optimizer], Optimizer]:
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("optimizer name must not be empty")

    def decorator(function: Optimizer) -> Optimizer:
        if normalized in _REGISTRY:
            raise ValueError(f"optimizer already registered: {normalized}")
        _REGISTRY[normalized] = function
        return function

    return decorator


def available_optimizers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _prepare_initial(initial: Sequence[str] | np.ndarray | None, space: SequenceSpace, config: OptimizationConfig, rng: np.random.Generator) -> np.ndarray:
    values = space.random(config.num_candidates, rng) if initial is None else space.validate(initial)
    if len(values) != config.num_candidates:
        raise ValueError(f"initial must contain exactly {config.num_candidates} sequences")
    if len(np.unique(values)) != len(values):
        raise ValueError("initial sequences must be unique")
    return values


def _finalize(name: str, archive: dict[str, tuple[float, float]], config: OptimizationConfig, query_count: int, metadata: dict[str, Any], trajectory: list[dict[str, float | int]]) -> OptimizationResult:
    if len(archive) < config.num_candidates:
        raise RuntimeError(f"only {len(archive)} unique sequences were scored; {config.num_candidates} required")
    ranked = sorted(archive.items(), key=lambda item: (-item[1][0], item[0]))[: config.num_candidates]
    return OptimizationResult(name, np.asarray([x[0] for x in ranked]), np.asarray([x[1][0] for x in ranked]), np.asarray([x[1][1] for x in ranked]), query_count, metadata, trajectory)


def _record(archive: dict[str, tuple[float, float]], sequences: np.ndarray, batch: ScoreBatch) -> None:
    uncertainties = np.zeros(len(batch.scores)) if batch.uncertainties is None else batch.uncertainties
    for sequence, score, uncertainty in zip(sequences, batch.scores, uncertainties):
        key = str(sequence)
        previous = archive.get(key)
        if previous is None or score > previous[0]:
            archive[key] = (float(score), float(uncertainty))


def _score(scorer: CachedScorer, archive: dict[str, tuple[float, float]], sequences: np.ndarray) -> ScoreBatch:
    batch = scorer.score(sequences)
    _record(archive, sequences, batch)
    return batch


@register_optimizer("random")
def random_search(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    _score(scorer, archive, initial)
    stagnant = 0
    while scorer.remaining is None or scorer.remaining > 0:
        count = min(4096, scorer.remaining or 4096)
        before = scorer.query_count
        _score(scorer, archive, space.random(count, rng))
        stagnant = stagnant + 1 if scorer.query_count == before else 0
        if stagnant >= 3:
            # The finite sequence space may be exhausted when unique-query
            # accounting is enabled; avoid an infinite loop in that case.
            break
    return _finalize("random", archive, config, scorer.query_count, {"strategy": "uniform random screening"}, [])


@register_optimizer("ism_greedy")
def ism_greedy(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    current = initial.copy()
    current_scores = _score(scorer, archive, current).scores
    trajectory = [{"query_count": scorer.query_count, "best_score": float(current_scores.max()), "mean_score": float(current_scores.mean())}]
    while scorer.remaining is None or scorer.remaining >= space.length * (space.alphabet_size - 1):
        before = scorer.query_count
        order = rng.permutation(len(current))
        max_seeds = len(order) if scorer.remaining is None else min(len(order), scorer.remaining // (space.length * (space.alphabet_size - 1)))
        chosen = order[:max_seeds]
        mutants = space.single_mutants(current[chosen])
        batch = _score(scorer, archive, mutants)
        block = space.length * (space.alphabet_size - 1)
        for offset, index in enumerate(chosen):
            values = batch.scores[offset * block : (offset + 1) * block]
            best = int(np.argmax(values))
            if values[best] > current_scores[index]:
                current[index] = mutants[offset * block + best]
                current_scores[index] = values[best]
        trajectory.append({"query_count": scorer.query_count, "best_score": float(current_scores.max()), "mean_score": float(current_scores.mean())})
        if not len(chosen):
            break
        if scorer.query_count == before:
            break
    return _finalize("ism_greedy", archive, config, scorer.query_count, {"sweeps": len(trajectory) - 1}, trajectory)


@register_optimizer("simulated_annealing")
def simulated_annealing(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    current = initial.copy(); current_scores = _score(scorer, archive, current).scores
    temperature = config.initial_temperature or max(float(np.std(current_scores)), 1e-3)
    trajectory=[]
    while scorer.remaining is None or scorer.remaining >= len(current):
        before = scorer.query_count
        proposals = space.mutate_one(current, rng); values = _score(scorer, archive, proposals).scores
        progress = scorer.query_count / max(config.budget, 1)
        temp = max(temperature * (0.001 ** progress), config.min_temperature)
        accept = (values >= current_scores) | (rng.random(len(current)) < np.exp(np.minimum((values - current_scores) / temp, 0)))
        current[accept] = proposals[accept]; current_scores[accept] = values[accept]
        trajectory.append({"query_count": scorer.query_count, "best_score": float(current_scores.max()), "mean_score": float(current_scores.mean())})
        if scorer.query_count == before:
            break
    return _finalize("simulated_annealing", archive, config, scorer.query_count, {"initial_temperature": temperature}, trajectory)


@register_optimizer("genetic_algorithm")
def genetic_algorithm(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    population = initial.copy(); scores = _score(scorer, archive, population).scores; elites = max(1, int(round(len(population) * config.elite_fraction))); trajectory=[]
    while scorer.remaining is None or scorer.remaining >= len(population):
        before = scorer.query_count
        elite = population[np.argsort(scores)[-elites:]]; a = space.encode(elite[rng.integers(0, len(elite), len(population))]); b = space.encode(elite[rng.integers(0, len(elite), len(population))]); child = np.where(rng.random(a.shape) < .5, a, b); mask = rng.random(child.shape) < config.mutation_rate; child = np.where(mask, rng.integers(0, space.alphabet_size, child.shape), child); children = space.decode(child); child_scores = _score(scorer, archive, children).scores; all_sequences=np.concatenate([population,children]); all_scores=np.concatenate([scores,child_scores]); keep=np.argsort(all_scores)[-len(population):]; population=all_sequences[keep]; scores=all_scores[keep]; trajectory.append({"query_count":scorer.query_count,"best_score":float(scores.max()),"mean_score":float(scores.mean())})
        if scorer.query_count == before:
            break
    return _finalize("genetic_algorithm", archive, config, scorer.query_count, {"generations": len(trajectory), "elite_count": elites}, trajectory)


@register_optimizer("adalead")
def adalead(initial, scorer, space, config, rng):
    # Generic AdaLead-style conservative hill climbing: continue a mutant only
    # when it beats its parent, and restart rejected branches from elites.
    archive: dict[str, tuple[float, float]] = {}; population=initial.copy(); scores=_score(scorer,archive,population).scores; trajectory=[]
    while scorer.remaining is None or scorer.remaining >= len(population):
        before = scorer.query_count
        elite_count=max(1,int(round(len(population)*config.elite_fraction))); elite_idx=np.argsort(scores)[-elite_count:]; parents=population[elite_idx[rng.integers(0,elite_count,len(population))]]; proposals=space.mutate_one(parents,rng); values=_score(scorer,archive,proposals).scores; improved=values>scores; population[improved]=proposals[improved]; scores[improved]=values[improved]; trajectory.append({"query_count":scorer.query_count,"best_score":float(scores.max()),"mean_score":float(scores.mean())})
        if scorer.query_count == before:
            break
    return _finalize("adalead", archive, config, scorer.query_count, {"steps": len(trajectory)}, trajectory)


def run_optimizer(name: str, initial: Sequence[str] | np.ndarray | None, scorer: SequenceScorer, space: SequenceSpace, config: OptimizationConfig, *, seed: int = 0, count_repeated: bool = False) -> OptimizationResult:
    normalized = str(name).strip().lower()
    if normalized not in _REGISTRY:
        raise KeyError(f"unknown optimizer {name!r}; available: {', '.join(available_optimizers())}")
    rng = np.random.default_rng(seed); random.seed(seed)
    prepared = _prepare_initial(initial, space, config, rng)
    budgeted = scorer if isinstance(scorer, CachedScorer) else CachedScorer(scorer, config.budget, count_repeated=count_repeated)
    if budgeted.budget is None:
        budgeted.budget = config.budget
    return _REGISTRY[normalized](prepared, budgeted, space, config, rng)


def run_optimizers(names: Iterable[str], initial: Sequence[str] | np.ndarray | None, scorer: SequenceScorer, space: SequenceSpace, config: OptimizationConfig, *, seed: int = 0) -> dict[str, OptimizationResult]:
    """Run multiple optimizers with independent scorers and reproducible seeds."""
    results: dict[str, OptimizationResult] = {}
    base_scorer = scorer.scorer if isinstance(scorer, CachedScorer) else scorer
    for offset, name in enumerate(names):
        # Each algorithm receives a fresh cache: a query budget measures that
        # algorithm's own model calls, not calls made by another algorithm.
        results[str(name)] = run_optimizer(name, initial, base_scorer, space, config, seed=seed + offset)
    return results
