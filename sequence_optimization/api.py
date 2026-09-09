"""Public data types and model adapters for sequence optimization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike


def normalize_nucleotide_sequences(
    sequences: Sequence[str] | np.ndarray,
) -> np.ndarray:
    """Normalize DNA/RNA strings to uppercase DNA notation (U becomes T)."""
    return np.char.replace(
        np.char.upper(np.asarray(sequences, dtype=str).reshape(-1)),
        "U",
        "T",
    )


def pad_nucleotide_sequences(
    sequences: Sequence[str] | np.ndarray,
    length: int,
    *,
    side: str = "right",
) -> np.ndarray:
    """Pad variable-length A/C/G/T sequences with N for fixed-input models."""
    if length < 1:
        raise ValueError("padding length must be positive")
    padding_side = str(side).strip().lower()
    if padding_side not in {"left", "right"}:
        raise ValueError("padding side must be 'left' or 'right'")
    values = normalize_nucleotide_sequences(sequences)
    lengths = np.char.str_len(values)
    if np.any(lengths > length):
        row = int(np.flatnonzero(lengths > length)[0])
        raise ValueError(
            f"sequence at row {row} has length {int(lengths[row])}, "
            f"exceeding model input length {length}"
        )
    if any(set(sequence) - set("ACGTN") for sequence in values):
        raise ValueError("sequences must use A/C/G/T/U or padding symbol N")
    if padding_side == "left":
        return np.asarray([sequence.rjust(length, "N") for sequence in values])
    return np.asarray([sequence.ljust(length, "N") for sequence in values])


def _prepare_model_sequences(values, length: int | None, side: str) -> np.ndarray:
    normalized = normalize_nucleotide_sequences(values)
    if length is None:
        return normalized
    return pad_nucleotide_sequences(normalized, length, side=side)


def _pad_soft_nucleotide_tensor(probabilities, length: int | None, side: str):
    if length is None:
        return probabilities
    if probabilities.ndim != 3:
        raise ValueError(
            "soft nucleotide input must have shape [batch, length, alphabet]"
        )
    difference = length - probabilities.shape[1]
    if difference < 0:
        raise ValueError(
            f"sequence length {probabilities.shape[1]} exceeds model input length {length}"
        )
    if difference == 0:
        return probabilities
    from torch.nn import functional as functional

    padding = (0, 0, difference, 0) if side == "left" else (0, 0, 0, difference)
    return functional.pad(probabilities, padding)


@dataclass(frozen=True)
class ScoreBatch:
    """One activity vector per sequence.

    ``scores`` may have shape ``[N]`` for a scalar model or ``[N, T]`` for a
    model with ``T`` activity outputs. Optimizers scalarize these activities
    with an objective before searching.
    """

    scores: np.ndarray
    uncertainties: np.ndarray | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=np.float64)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        if scores.ndim not in {1, 2}:
            raise ValueError("scores must have shape [N] or [N, tasks]")
        if not np.isfinite(scores).all():
            raise ValueError("scorer returned non-finite scores")
        uncertainty = self.uncertainties
        if uncertainty is not None:
            uncertainty = np.asarray(uncertainty, dtype=np.float64)
            if uncertainty.ndim == 0:
                uncertainty = uncertainty.reshape(1)
            if uncertainty.shape != scores.shape:
                raise ValueError("uncertainties must have the same shape as scores")
            if not np.isfinite(uncertainty).all():
                raise ValueError("scorer returned non-finite uncertainties")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "uncertainties", uncertainty)


ScorerOutput: TypeAlias = (
    ScoreBatch | ArrayLike | tuple[ArrayLike, ArrayLike | None]
)


def as_score_batch(result: ScorerOutput) -> ScoreBatch:
    """Normalize a scorer result to the optimizer's internal representation.

    External model packages may return activity arrays directly, while native
    adapters can return :class:`ScoreBatch`. A two-item tuple is interpreted as
    ``(activities, uncertainties)``.
    """
    if isinstance(result, ScoreBatch):
        return result

    def to_numpy(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return value

    if isinstance(result, tuple) and len(result) == 2:
        scores, uncertainties = result
        return ScoreBatch(
            to_numpy(scores),
            None if uncertainties is None else to_numpy(uncertainties),
        )
    return ScoreBatch(to_numpy(result))


class SequenceScorer(Protocol):
    """A model shell: sequence strings in, one or more activities out."""

    def score(self, sequences: Sequence[str] | np.ndarray) -> ScorerOutput: ...


class CallableScorer:
    """Adapt a callable returning scores, ``(scores, uncertainty)``, or ScoreBatch."""

    def __init__(
        self,
        function: Callable[[np.ndarray], Any],
        *,
        task_names: Sequence[str] | None = None,
        pad_to_length: int | None = None,
        padding_side: str = "right",
    ) -> None:
        self.function = function
        self.task_names = _validate_task_names(task_names)
        self.pad_to_length, self.padding_side = _validate_padding(
            pad_to_length, padding_side
        )

    def score(self, sequences: Sequence[str] | np.ndarray) -> ScoreBatch:
        values = _prepare_model_sequences(
            sequences, self.pad_to_length, self.padding_side
        )
        return as_score_batch(self.function(values))


class TorchSequenceScorer:
    """Wrap any frozen PyTorch model with a user-supplied string encoder."""

    def __init__(
        self,
        model: Any,
        encoder: Callable[[np.ndarray], Any],
        *,
        device: Any = None,
        batch_size: int = 4096,
        task_names: Sequence[str] | None = None,
        pad_to_length: int | None = None,
        padding_side: str = "right",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model, self.encoder, self.device, self.batch_size = (
            model,
            encoder,
            device,
            batch_size,
        )
        self.task_names = _validate_task_names(task_names)
        self.pad_to_length, self.padding_side = _validate_padding(
            pad_to_length, padding_side
        )
        if device is not None and hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()

    def score(self, sequences: Sequence[str] | np.ndarray) -> ScoreBatch:
        import torch

        values = _prepare_model_sequences(
            sequences, self.pad_to_length, self.padding_side
        )
        scores, uncertainties = [], []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                inputs = self.encoder(batch)
                if self.device is not None and hasattr(inputs, "to"):
                    inputs = inputs.to(self.device)
                output = self.model(inputs)
                current = as_score_batch(output)
                if len(current.scores) != len(batch):
                    raise ValueError(
                        f"model returned {len(current.scores)} rows for a batch of {len(batch)} sequences"
                    )
                scores.append(current.scores)
                uncertainties.append(
                    np.zeros_like(current.scores)
                    if current.uncertainties is None
                    else current.uncertainties
                )
        combined = np.concatenate(scores, axis=0) if scores else np.empty(0)
        combined_uncertainty = (
            np.concatenate(uncertainties, axis=0)
            if uncertainties
            else np.zeros_like(combined)
        )
        return ScoreBatch(combined, combined_uncertainty)


class DifferentiableTorchSequenceScorer(TorchSequenceScorer):
    """PyTorch scorer that also accepts differentiable soft one-hot sequences.

    Discrete optimizers use :meth:`score`. Gradient-based optimizers call
    :meth:`score_soft` with tensors shaped ``[batch, length, alphabet]``. The
    optional transform adapts that canonical representation to the wrapped
    model's expected tensor layout.
    """

    def __init__(
        self,
        model: Any,
        encoder: Callable[[np.ndarray], Any],
        *,
        soft_input_transform: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, encoder, **kwargs)
        self.soft_input_transform = soft_input_transform or (lambda values: values)
        if hasattr(model, "parameters"):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def score_soft(self, probabilities: Any) -> tuple[Any, Any | None]:
        """Return differentiable activities and optional uncertainties."""
        probabilities = _pad_soft_nucleotide_tensor(
            probabilities, self.pad_to_length, self.padding_side
        )
        inputs = self.soft_input_transform(probabilities)
        if self.device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        output = self.model(inputs)
        if isinstance(output, tuple) and len(output) == 2:
            activities, uncertainties = output
        else:
            activities, uncertainties = output, None
        if not hasattr(activities, "ndim"):
            raise TypeError("differentiable model output must be a tensor")
        if activities.ndim == 1:
            activities = activities[:, None]
        if activities.ndim != 2 or len(activities) != len(probabilities):
            raise ValueError("differentiable model must return shape [N] or [N, tasks]")
        if uncertainties is not None:
            if uncertainties.ndim == 1:
                uncertainties = uncertainties[:, None]
            if uncertainties.shape != activities.shape:
                raise ValueError("differentiable uncertainties must match activities")
        return activities, uncertainties


def _validate_task_names(task_names: Sequence[str] | None) -> tuple[str, ...] | None:
    if task_names is None:
        return None
    result = tuple(str(name).strip() for name in task_names)
    if (
        not result
        or any(not name for name in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError("task_names must contain unique, non-empty names")
    return result


def _validate_padding(length: int | None, side: str) -> tuple[int | None, str]:
    if length is not None and length < 1:
        raise ValueError("pad_to_length must be positive or None")
    padding_side = str(side).strip().lower()
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    return length, padding_side


class BudgetExceeded(RuntimeError):
    """The optimizer requested more evaluations than its configured budget."""


class CachedScorer:
    """Add duplicate caching and query accounting to any scorer."""

    def __init__(
        self,
        scorer: SequenceScorer,
        budget: int | None = None,
        *,
        count_repeated: bool = False,
    ) -> None:
        if budget is not None and budget < 0:
            raise ValueError("budget must be non-negative or None")
        self.scorer, self.budget, self.count_repeated = scorer, budget, count_repeated
        self.query_count = 0
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def remaining(self) -> int | None:
        return None if self.budget is None else self.budget - self.query_count

    def score(self, sequences: Sequence[str] | np.ndarray) -> ScoreBatch:
        values = normalize_nucleotide_sequences(sequences)
        if not len(values):
            return ScoreBatch(np.empty(0), np.empty(0))
        missing = list(
            dict.fromkeys(
                str(value) for value in values if str(value) not in self._cache
            )
        )
        cost = len(values) if self.count_repeated else len(missing)
        if self.budget is not None and cost > self.remaining:
            raise BudgetExceeded(
                f"query budget exceeded: requested {cost}, remaining {self.remaining}"
            )
        if missing:
            current = as_score_batch(
                self.scorer.score(np.asarray(missing, dtype=str))
            )
            if len(current.scores) != len(missing):
                raise ValueError(
                    f"scorer returned {len(current.scores)} rows for {len(missing)} sequences"
                )
            uncertainty = (
                np.zeros_like(current.scores)
                if current.uncertainties is None
                else current.uncertainties
            )
            self._cache.update(
                {
                    key: (
                        np.asarray(score, dtype=np.float64).copy(),
                        np.asarray(std, dtype=np.float64).copy(),
                    )
                    for key, score, std in zip(missing, current.scores, uncertainty)
                }
            )
        self.query_count += cost
        return ScoreBatch(
            np.stack([self._cache[str(value)][0] for value in values]),
            np.stack([self._cache[str(value)][1] for value in values]),
        )


@dataclass(frozen=True)
class SequenceSpace:
    length: int
    alphabet: str = "ACGT"

    def __post_init__(self) -> None:
        alphabet = str(self.alphabet).upper().replace("U", "T")
        object.__setattr__(self, "alphabet", alphabet)
        if self.length < 1 or len(alphabet) < 2 or len(set(alphabet)) != len(alphabet):
            raise ValueError(
                "length must be positive and alphabet must contain unique symbols"
            )

    @property
    def alphabet_size(self) -> int:
        return len(self.alphabet)

    def validate(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        values = normalize_nucleotide_sequences(sequences)
        if np.any(np.char.str_len(values) != self.length):
            raise ValueError(f"all sequences must have length {self.length}")
        allowed = set(self.alphabet)
        if any(set(value) - allowed for value in values):
            raise ValueError(f"sequences must use alphabet {self.alphabet!r}")
        return values

    def encode(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        values = self.validate(sequences)
        lookup = {symbol: index for index, symbol in enumerate(self.alphabet)}
        return np.asarray(
            [[lookup[symbol] for symbol in value] for value in values], dtype=np.int16
        )

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        values = np.asarray(encoded, dtype=np.int64)
        if (
            values.ndim != 2
            or values.shape[1] != self.length
            or np.any(values < 0)
            or np.any(values >= self.alphabet_size)
        ):
            raise ValueError("invalid encoded sequence array")
        symbols = np.asarray(list(self.alphabet))
        return np.asarray(["".join(symbols[row]) for row in values], dtype=str)

    def random(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return self.decode(rng.integers(self.alphabet_size, size=(count, self.length)))

    def single_mutants(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        encoded = self.encode(sequences)
        blocks = []
        for sequence in encoded:
            candidates = []
            for position in range(self.length):
                for value in range(self.alphabet_size):
                    if value != sequence[position]:
                        candidate = sequence.copy()
                        candidate[position] = value
                        candidates.append(candidate)
            blocks.append(np.asarray(candidates, dtype=np.int16))
        return self.decode(np.asarray(blocks).reshape(-1, self.length))

    def mutate_one(
        self, sequences: Sequence[str] | np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        encoded = self.encode(sequences).copy()
        rows = np.arange(len(encoded))
        positions = rng.integers(self.length, size=len(encoded))
        offsets = rng.integers(1, self.alphabet_size, size=len(encoded))
        encoded[rows, positions] = (
            encoded[rows, positions] + offsets
        ) % self.alphabet_size
        return self.decode(encoded)


@dataclass(frozen=True)
class OptimizationConfig:
    budget: int
    num_candidates: int = 100
    mutation_rate: float = 0.02
    elite_fraction: float = 0.2
    learning_rate: float = 0.1
    generator_learning_rate: float = 3e-3
    generator_pretrain_epochs: int = 20
    generator_update_epochs: int = 5
    cbas_quantile: float = 0.8
    cbas_batch_size: int = 256
    den_seed_pretrain_epochs: int = 100
    den_diversity_weight: float = 0.1
    ledidi_edit_penalty: float = 0.1
    ledidi_temperature: float = 1.0
    ledidi_epsilon: float = 1e-4
    basesearch_decay: float = 0.995
    basesearch_queue_size: int = 1024
    basesearch_max_stalled: int = 1024

    def __post_init__(self) -> None:
        if not 0 < self.basesearch_decay < 1:
            raise ValueError("basesearch_decay must be between zero and one")
        for name in ("basesearch_queue_size", "basesearch_max_stalled"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.ledidi_edit_penalty) or self.ledidi_edit_penalty < 0:
            raise ValueError("ledidi_edit_penalty must be finite and non-negative")
        for name in ("ledidi_temperature", "ledidi_epsilon"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.budget < self.num_candidates or self.num_candidates < 1:
            raise ValueError("budget must be at least num_candidates")
        if (
            not 0 < self.mutation_rate <= 1
            or not 0 < self.elite_fraction <= 1
            or self.learning_rate <= 0
        ):
            raise ValueError("invalid optimizer hyperparameters")
        if self.generator_learning_rate <= 0:
            raise ValueError("generator_learning_rate must be positive")
        if self.generator_pretrain_epochs < 0 or self.generator_update_epochs < 0:
            raise ValueError("generator epoch counts must be non-negative")
        if not 0 < self.cbas_quantile < 1 or self.cbas_batch_size < 1:
            raise ValueError("invalid CbAS hyperparameters")
        if self.den_seed_pretrain_epochs < 0 or self.den_diversity_weight < 0:
            raise ValueError("invalid DEN hyperparameters")


@dataclass
class OptimizationResult:
    algorithm: str
    sequences: np.ndarray
    scores: np.ndarray
    uncertainties: np.ndarray
    query_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, float | int]] = field(default_factory=list)
    activities: np.ndarray | None = None
    task_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.sequences = np.asarray(self.sequences, dtype=str).reshape(-1)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        self.uncertainties = np.asarray(self.uncertainties, dtype=np.float64).reshape(
            -1
        )
        if not (len(self.sequences) == len(self.scores) == len(self.uncertainties)):
            raise ValueError("result arrays must have equal lengths")
        if len(np.unique(self.sequences)) != len(self.sequences):
            raise ValueError("result sequences must be unique")
        self.task_names = (
            _validate_task_names(self.task_names) if self.task_names else ()
        )
        self._validate_activities()

    def _validate_activities(self) -> None:
        if self.activities is not None:
            activities = np.asarray(self.activities, dtype=np.float64)
            if activities.ndim == 1:
                activities = activities[:, None]
            if activities.ndim != 2 or len(activities) != len(self.sequences):
                raise ValueError("activities must have shape [num_candidates, tasks]")
            self.activities = activities
            if self.task_names and len(self.task_names) != activities.shape[1]:
                raise ValueError("task_names must match the activity columns")
            if not self.task_names:
                self.task_names = tuple(
                    f"task_{index}" for index in range(activities.shape[1])
                )

    def attach_activities(
        self,
        activities: np.ndarray,
        task_names: Sequence[str] | None = None,
    ) -> "OptimizationResult":
        """Attach the raw model outputs retained during scalar optimization."""
        self.activities = activities
        self.task_names = _validate_task_names(task_names) if task_names else ()
        self._validate_activities()
        return self

    def to_frame(self):
        import pandas as pd

        frame = pd.DataFrame(
            {
                "sequence": self.sequences,
                "score": self.scores,
                "uncertainty": self.uncertainties,
            }
        )
        if self.activities is not None:
            for index, name in enumerate(self.task_names):
                frame[f"activity_{name}"] = self.activities[:, index]
        evaluation = self.metadata.get("objective_evaluation")
        if evaluation is not None:
            frame["worst_margin"] = evaluation["worst_margin"]
            for index, name in enumerate(evaluation["task_names"]):
                frame[f"margin_{name}"] = np.asarray(evaluation["margins"])[:, index]
                if evaluation["satisfied"] is not None:
                    frame[f"satisfied_{name}"] = np.asarray(evaluation["satisfied"])[:, index]
            if evaluation["all_satisfied"] is not None:
                frame["all_satisfied"] = evaluation["all_satisfied"]
        return frame
