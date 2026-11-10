"""Public data types and model adapters for sequence optimization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

try:
    from ._sequence_kernels import single_mutants as _native_single_mutants
except ImportError:
    _native_single_mutants = None


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


ScorerOutput: TypeAlias = ScoreBatch | ArrayLike | tuple[ArrayLike, ArrayLike | None]
FixedRegionSpec: TypeAlias = int | str | slice


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


class SequencePredictor:
    """Wrap a PyTorch model for discrete and gradient sequence optimization.

    The default model input is float32 one-hot ``[batch, 4, length]`` in A/C/G/T
    order. ``input_transform`` maps canonical probabilities to a model-specific
    input; ``output_transform`` extracts or constructs activity tensors.
    ``score`` receives nucleotide strings, while ``score_soft`` receives the
    optimizer's canonical ``[batch, length, A/C/G/T]`` probabilities.
    Both scoring paths return activities with shape ``[batch, tasks]``.
    """

    supports_soft_scoring = True

    def __init__(
        self,
        model: Any,
        *,
        device: Any = None,
        batch_size: int = 4096,
        tasks: Sequence[str] | None = None,
        pad_to_length: int | None = None,
        padding_side: str = "right",
        task_names: Sequence[str] | None = None,
        input_transform: Callable[[Any], Any] | None = None,
        output_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if task_names is not None:
            # Deprecated alias retained for callers written against the old name.
            if tasks is not None:
                raise TypeError("pass tasks or task_names, not both")
            tasks = task_names
        self.model, self.device, self.batch_size = (
            model,
            device,
            batch_size,
        )
        self.input_transform = input_transform
        self.output_transform = output_transform
        self._user_tasks = tasks  # Store user-provided task names
        self.task_names = _validate_task_names(tasks)  # Will be updated after first prediction
        self.pad_to_length, self.padding_side = _validate_padding(
            pad_to_length, padding_side
        )
        if device is not None and hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "parameters"):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def _infer_task_names(self, num_tasks: int) -> tuple[str, ...]:
        """Generate default task names if not provided by user."""
        if self._user_tasks is not None:
            return self.task_names  # Already validated
        return tuple(f"task_{i}" for i in range(num_tasks))

    def encode_sequences(self, sequences: Sequence[str] | np.ndarray) -> Any:
        """Encode strings as canonical float32 ``[batch, length, 4]`` tensors."""
        import torch

        from .onehot import onehot

        return torch.from_numpy(onehot(sequences))

    def prepare_input(self, probabilities: Any) -> Any:
        """Convert canonical ``[batch, length, 4]`` values for the model."""
        if self.input_transform is not None:
            return self.input_transform(probabilities)
        return probabilities.transpose(1, 2)

    def score(self, sequences: Sequence[str] | np.ndarray) -> ScoreBatch:
        import torch

        values = _prepare_model_sequences(
            sequences, self.pad_to_length, self.padding_side
        )
        scores, uncertainties = [], []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                encoded = self.encode_sequences(batch)
                if self.device is not None and hasattr(encoded, "to"):
                    encoded = encoded.to(self.device)
                inputs = self.prepare_input(encoded)
                if isinstance(inputs, np.ndarray):
                    inputs = torch.from_numpy(inputs)
                if self.device is not None and hasattr(inputs, "to"):
                    inputs = inputs.to(self.device)
                output = self.model(inputs)
                if self.output_transform is not None:
                    output = self.output_transform(output)
                current = as_score_batch(output)
                if current.scores.ndim == 1:
                    current = ScoreBatch(
                        current.scores[:, None],
                        None if current.uncertainties is None else current.uncertainties[:, None],
                    )
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

        # Infer task names from first prediction if not provided by user
        if self._user_tasks is None and combined.ndim > 1:
            num_tasks = combined.shape[1]
            self.task_names = self._infer_task_names(num_tasks)

        return ScoreBatch(combined, combined_uncertainty)

    def score_soft(self, probabilities: Any) -> tuple[Any, Any | None]:
        """Return differentiable activities and optional uncertainties."""
        probabilities = _pad_soft_nucleotide_tensor(
            probabilities, self.pad_to_length, self.padding_side
        )
        if self.device is not None:
            probabilities = probabilities.to(self.device)
        inputs = self.prepare_input(probabilities)
        if self.device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        output = self.model(inputs)
        if self.output_transform is not None:
            output = self.output_transform(output)
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

        # Infer task names from first prediction if not provided by user
        if self._user_tasks is None and activities.shape[1] > 0:
            num_tasks = activities.shape[1]
            self.task_names = self._infer_task_names(num_tasks)

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
        constraint=None,
        constraint_rng: np.random.Generator | None = None,
    ) -> None:
        if budget is not None and budget < 0:
            raise ValueError("budget must be non-negative or None")
        self.scorer, self.budget, self.count_repeated = scorer, budget, count_repeated
        self.query_count = 0
        self.model_query_count = 0
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._rejected: set[str] = set()
        self.constraint_evaluator = None
        if constraint is not None:
            from .constraints import ConstraintEvaluator

            self.constraint_evaluator = ConstraintEvaluator(
                constraint,
                np.random.default_rng() if constraint_rng is None else constraint_rng,
            )

    def set_constraint(
        self, constraint, rng: np.random.Generator | None = None
    ) -> None:
        """Set a constraint before any values have been cached."""
        if self._cache or self._rejected:
            raise ValueError(
                "constraint cannot change after sequences have been cached"
            )
        if constraint is None:
            self.constraint_evaluator = None
            return
        from .constraints import ConstraintEvaluator

        self.constraint_evaluator = ConstraintEvaluator(
            constraint, np.random.default_rng() if rng is None else rng
        )

    @property
    def remaining(self) -> int | None:
        if bool(getattr(self.scorer, "target_stop_reached", False)):
            return 0
        return None if self.budget is None else self.budget - self.query_count

    @property
    def objective_version(self) -> int:
        return int(getattr(self.scorer, "objective_version", 0))

    def current_scores(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        """Return scores under the current objective state without spending budget."""
        values = normalize_nucleotide_sequences(sequences)
        if not len(values):
            return np.empty(0, dtype=np.float64)
        rescore = getattr(self.scorer, "scores_for", None)
        if rescore is not None:
            return np.asarray(rescore(values), dtype=np.float64)
        missing = [str(value) for value in values if str(value) not in self._cache]
        if missing:
            raise KeyError(f"scores were not cached for {len(missing)} sequences")
        return np.asarray([self._cache[str(value)][0] for value in values])

    def current_uncertainties(
        self, sequences: Sequence[str] | np.ndarray
    ) -> np.ndarray | None:
        """Return uncertainties under the current objective state."""
        values = normalize_nucleotide_sequences(sequences)
        if not len(values):
            return np.empty(0, dtype=np.float64)
        rescore = getattr(self.scorer, "uncertainties_for", None)
        if rescore is not None:
            current = rescore(values)
            if current is not None:
                return np.asarray(current, dtype=np.float64)
        missing = [str(value) for value in values if str(value) not in self._cache]
        if missing:
            raise KeyError(f"scores were not cached for {len(missing)} sequences")
        return np.asarray([self._cache[str(value)][1] for value in values])

    def score(self, sequences: Sequence[str] | np.ndarray) -> ScoreBatch:
        values = normalize_nucleotide_sequences(sequences)
        if not len(values):
            return ScoreBatch(np.empty(0), np.empty(0))
        missing = list(
            dict.fromkeys(
                str(value)
                for value in values
                if str(value) not in self._cache and str(value) not in self._rejected
            )
        )
        cost = len(values) if self.count_repeated else len(missing)
        if self.budget is not None and cost > self.remaining:
            raise BudgetExceeded(
                f"query budget exceeded: requested {cost}, remaining {self.remaining}"
            )
        if missing:
            accepted = missing
            if self.constraint_evaluator is not None:
                passed = self.constraint_evaluator.evaluate(missing)
                accepted = [value for value, keep in zip(missing, passed) if keep]
                self._rejected.update(
                    value for value, keep in zip(missing, passed) if not keep
                )
            current = (
                as_score_batch(self.scorer.score(np.asarray(accepted, dtype=str)))
                if accepted
                else ScoreBatch(np.empty(0), np.empty(0))
            )
            if len(current.scores) != len(accepted):
                raise ValueError(
                    f"scorer returned {len(current.scores)} rows for {len(accepted)} sequences"
                )
            if self.constraint_evaluator is not None and current.scores.ndim != 1:
                raise ValueError(
                    "constraints require scalar objective scores; pass objective to the optimizer"
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
                    for key, score, std in zip(accepted, current.scores, uncertainty)
                }
            )
            self.model_query_count += len(accepted)
        self.query_count += cost
        rejected_score = np.asarray(np.finfo(np.float64).min)
        scores = np.stack(
            [
                rejected_score
                if str(value) in self._rejected
                else self._cache[str(value)][0]
                for value in values
            ]
        )
        accepted_mask = np.asarray(
            [str(value) not in self._rejected for value in values], dtype=bool
        )
        if accepted_mask.any() and hasattr(self.scorer, "scores_for"):
            scores = np.asarray(scores, dtype=np.float64)
            scores[accepted_mask] = self.scorer.scores_for(values[accepted_mask])
        uncertainties = np.stack(
            [
                np.zeros_like(rejected_score)
                if str(value) in self._rejected
                else self._cache[str(value)][1]
                for value in values
            ]
        )
        if accepted_mask.any() and hasattr(self.scorer, "uncertainties_for"):
            current = self.scorer.uncertainties_for(values[accepted_mask])
            if current is not None:
                uncertainties = np.asarray(uncertainties, dtype=np.float64)
                uncertainties[accepted_mask] = current
        return ScoreBatch(scores, uncertainties)

    def constraints_passed(self, sequences) -> np.ndarray:
        values = normalize_nucleotide_sequences(sequences)
        passed = self.hard_constraints_passed(values)
        threshold_filter = getattr(self.scorer, "thresholds_passed", None)
        if threshold_filter is not None:
            passed &= np.asarray(threshold_filter(values), dtype=bool)
        return passed

    def hard_constraints_passed(self, sequences) -> np.ndarray:
        """Return user-constraint decisions without objective target gates."""
        if self.constraint_evaluator is None:
            return np.ones(len(sequences), dtype=bool)
        values = normalize_nucleotide_sequences(sequences)
        return (
            np.ones(len(values), dtype=bool)
            if self.constraint_evaluator is None
            else self.constraint_evaluator.evaluate(values)
        )


@dataclass(frozen=True)
class SequenceSpace:
    length: int
    alphabet: str = "ACGT"

    def __post_init__(self) -> None:
        alphabet = str(self.alphabet).upper().replace("U", "T")
        object.__setattr__(self, "alphabet", alphabet)
        if isinstance(self.length, bool) or not isinstance(
            self.length, (int, np.integer)
        ):
            raise ValueError("length must be a positive integer")
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
        if _native_single_mutants is not None and self.alphabet.isascii():
            values = self.validate(sequences)
            return np.asarray(_native_single_mutants(values, self.alphabet, tuple(range(self.length))), dtype=str)
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


def resolve_fixed_positions(
    fixed_region: Sequence[FixedRegionSpec] | None,
    length: int,
) -> tuple[int, ...]:
    """Resolve integer and Python-style slice entries to zero-based positions."""
    if fixed_region is None:
        return ()
    if isinstance(fixed_region, (str, bytes)) or not isinstance(fixed_region, Sequence):
        raise ValueError("fixed_region must be a list of positions or slices")
    positions: set[int] = set()
    for entry in fixed_region:
        region: int | slice
        if isinstance(entry, bool):
            raise ValueError("fixed_region entries must be integers or slices")
        if isinstance(entry, (int, np.integer)):
            region = int(entry)
        elif isinstance(entry, str):
            value = entry.strip()
            if not value:
                raise ValueError("fixed_region entries must not be empty")
            if ":" not in value:
                try:
                    region = int(value)
                except ValueError as error:
                    raise ValueError(f"invalid fixed_region entry {entry!r}") from error
            else:
                parts = value.split(":")
                if len(parts) != 2:
                    raise ValueError("fixed_region string slices must use start:stop")
                try:
                    start, stop = (
                        None if part.strip() == "" else int(part) for part in parts
                    )
                except ValueError as error:
                    raise ValueError(f"invalid fixed_region entry {entry!r}") from error
                region = slice(start, stop)
        elif isinstance(entry, slice):
            if entry.step not in {None, 1}:
                raise ValueError("fixed_region slices do not support a step")
            region = entry
        elif isinstance(entry, (Sequence, np.ndarray)):
            # [start, stop] interval pairs, with Python slice semantics.
            bounds = list(entry)
            if len(bounds) != 2:
                raise ValueError(
                    "fixed_region interval entries must contain exactly "
                    "[start, stop]"
                )
            parsed: list[int | None] = []
            for bound in bounds:
                if bound is None:
                    parsed.append(None)
                elif isinstance(bound, bool) or not isinstance(
                    bound, (int, np.integer)
                ):
                    raise ValueError(
                        "fixed_region interval bounds must be integers or None"
                    )
                else:
                    parsed.append(int(bound))
            region = slice(parsed[0], parsed[1])
        else:
            raise ValueError("fixed_region entries must be integers or slices")

        if isinstance(region, int):
            position = region + length if region < 0 else region
            if not 0 <= position < length:
                raise ValueError(
                    f"fixed_region position {region} is outside length {length}"
                )
            current = (position,)
        else:
            start, stop, _ = region.indices(length)
            current = tuple(range(start, stop))
            if not current:
                raise ValueError(
                    f"fixed_region slice {region.start}:{region.stop} is empty "
                    f"for length {length}"
                )
        positions.update(current)
    return tuple(sorted(positions))


class FixedRegionSequenceSpace:
    """Restrict a sequence space to one shared template at selected positions."""

    def __init__(self, base: SequenceSpace, template: str, positions: Sequence[int]):
        self.base = base
        self.length = base.length
        self.alphabet = base.alphabet
        self.alphabet_size = base.alphabet_size
        self.template = str(base.validate([template])[0])
        self.fixed_positions = tuple(int(position) for position in positions)
        fixed = set(self.fixed_positions)
        self.mutable_positions = tuple(
            position for position in range(self.length) if position not in fixed
        )
        self.max_hamming_distance = len(self.mutable_positions)
        self._template_encoded = base.encode([self.template])[0]
        if hasattr(base, "choices"):
            self.capacity = int(getattr(base, "capacity"))
        else:
            self.capacity = self.alphabet_size ** len(self.mutable_positions)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _matches(self, sequence: str) -> bool:
        return all(
            sequence[position] == self.template[position]
            for position in self.fixed_positions
        )

    def validate(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        values = self.base.validate(sequences)
        if any(not self._matches(str(value)) for value in values):
            raise ValueError(
                "all initial sequences must agree at fixed_region positions"
            )
        return values

    def encode(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        return self.base.encode(self.validate(sequences))

    def constrain_encoded(self, encoded):
        constrain = getattr(self.base, "constrain_encoded", None)
        if constrain is not None:
            encoded = constrain(encoded)
        values = (
            encoded.clone()
            if hasattr(encoded, "clone")
            else np.array(encoded, copy=True)
        )
        if not self.fixed_positions:
            return values
        if hasattr(values, "device"):
            import torch

            positions = torch.as_tensor(
                self.fixed_positions, dtype=torch.long, device=values.device
            )
            template = torch.as_tensor(
                self._template_encoded[list(self.fixed_positions)],
                dtype=values.dtype,
                device=values.device,
            )
            values[..., positions] = template
        else:
            values[..., list(self.fixed_positions)] = self._template_encoded[
                list(self.fixed_positions)
            ]
        return values

    def constrain_probabilities(self, probabilities):
        constrain = getattr(self.base, "constrain_probabilities", None)
        if constrain is not None:
            probabilities = constrain(probabilities)
        if not self.fixed_positions:
            return probabilities
        import torch
        from torch.nn import functional as functional

        template = torch.as_tensor(
            self._template_encoded,
            dtype=torch.long,
            device=probabilities.device,
        )
        fixed_values = functional.one_hot(template, num_classes=self.alphabet_size).to(
            probabilities.dtype
        )
        mask = torch.zeros(self.length, dtype=torch.bool, device=probabilities.device)
        mask[list(self.fixed_positions)] = True
        return torch.where(mask[None, :, None], fixed_values[None, :, :], probabilities)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        return self.validate(self.base.decode(self.constrain_encoded(encoded)))

    def random(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if hasattr(self.base, "choices"):
            # Codon spaces must apply the fixed bases after sampling so that
            # candidates remain valid. Resample candidates that become
            # unchanged CDS references after the fixed positions are applied.
            accepted = []
            maximum_draws = max(1_000, 100 * count)
            draws = 0
            reference = getattr(self.base, "initial_cds", None)
            while len(accepted) < count and draws < maximum_draws:
                batch_size = max(64, count - len(accepted))
                candidates = np.asarray(self.base.random(batch_size, rng), dtype=str).reshape(
                    -1
                )
                draws += len(candidates)
                if self.fixed_positions:
                    template = str(self.template)
                    candidates = np.asarray(
                        [
                            "".join(
                                template[position] if position in self.fixed_positions else base
                                for position, base in enumerate(value)
                            )
                            for value in candidates
                        ],
                        dtype=str,
                    )
                candidates = self.validate(candidates)
                if reference is not None:
                    codon_count = len(reference) // 3
                    changes = np.asarray(
                        [
                            sum(
                                candidate[index : index + 3]
                                != reference[index : index + 3]
                                for index in range(0, len(reference), 3)
                            )
                            for candidate in candidates
                        ]
                    )
                    candidates = candidates[(changes > 0) & (changes < codon_count - 1)]
                accepted.extend(candidates.tolist())
            if len(accepted) < count:
                raise ValueError(
                    "fixed_region leaves too few valid synonymous CDS variants "
                    "for random search"
                )
            return np.asarray(accepted[:count], dtype=str)
        encoded = rng.integers(
            self.alphabet_size, size=(count, self.length), dtype=np.int16
        )
        return self.decode(encoded)

    def single_mutants(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        if hasattr(self.base, "choices"):
            return self.validate(self.base.single_mutants(sequences))
        if _native_single_mutants is not None and self.alphabet.isascii():
            values = self.validate(sequences)
            return np.asarray(_native_single_mutants(values, self.alphabet, self.mutable_positions), dtype=str)
        encoded = self.encode(sequences)
        candidates = []
        for sequence in encoded:
            for position in self.mutable_positions:
                for value in range(self.alphabet_size):
                    if value != sequence[position]:
                        candidate = sequence.copy()
                        candidate[position] = value
                        candidates.append(candidate)
        if not candidates:
            return np.empty(0, dtype=str)
        return self.decode(np.asarray(candidates, dtype=np.int16))

    def mutate_one(
        self, sequences: Sequence[str] | np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if hasattr(self.base, "choices"):
            return self.validate(self.base.mutate_one(sequences, rng))
        encoded = self.encode(sequences).copy()
        if not self.mutable_positions:
            return self.decode(encoded)
        rows = np.arange(len(encoded))
        positions = rng.choice(self.mutable_positions, size=len(encoded))
        offsets = rng.integers(1, self.alphabet_size, size=len(encoded))
        encoded[rows, positions] = (
            encoded[rows, positions] + offsets
        ) % self.alphabet_size
        return self.decode(encoded)

    def offspring(self, elite, count, mutation_rate, rng):
        if hasattr(self.base, "offspring"):
            return self.validate(self.base.offspring(elite, count, mutation_rate, rng))
        a = self.encode(elite[rng.integers(len(elite), size=count)])
        b = self.encode(elite[rng.integers(len(elite), size=count)])
        child = np.where(rng.random(a.shape) < 0.5, a, b)
        mask = rng.random(child.shape) < mutation_rate
        if self.fixed_positions:
            mask[:, list(self.fixed_positions)] = False
        child = np.where(
            mask, rng.integers(self.alphabet_size, size=child.shape), child
        )
        return self.decode(child)


def _sequence_space(length, alphabet="ACGT") -> SequenceSpace:
    """Construct internal sequence operations; retain positional legacy objects."""
    if isinstance(length, SequenceSpace):
        if alphabet != "ACGT" and alphabet != length.alphabet:
            raise ValueError("alphabet conflicts with the supplied sequence space")
        return length
    return SequenceSpace(length, alphabet)


@dataclass(frozen=True)
class OptimizationConfig:
    budget: int
    num_candidates: int = 100
    diversity: int = 5
    mutation_rate: float = 0.02
    elite_fraction: float = 0.2
    parent_pool_multiplier: int = 5
    adalead_mu: float = 1.0
    adalead_max_rollout: int = 8
    adalead_allow_equal: bool = True
    adalead_lineages: int | None = None
    adalead_exploration_fraction: float = 0.2
    adalead_quality_power: float = 4.0
    ism_lineage_stopping: bool = True
    ism_min_sweeps: int = 10
    ism_patience: int = 5
    ism_absolute_gain: float = 0.01
    ism_relative_gain: float = 0.001
    ism_knee_slope_ratio: float = 0.1
    annealing_iterations: int = 1000
    annealing_initial_temperature: float = 0.1
    annealing_final_temperature: float = 0.05
    learning_rate: float = 0.1
    generator_learning_rate: float = 3e-3
    generator_pretrain_epochs: int = 20
    generator_update_epochs: int = 5
    cross_entropy_batch_size: int = 256
    den_seed_pretrain_epochs: int = 100
    den_diversity_weight: float = 0.1
    ledidi_edit_penalty: float = 0.1
    ledidi_temperature: float = 1.0
    ledidi_epsilon: float = 1e-4
    basesearch_decay: float = 0.995
    basesearch_queue_size: int = 1024
    basesearch_max_stalled: int = 1024
    basesearch_batch_size: int = 256
    fixed_region: Sequence[FixedRegionSpec] | None = None
    is_CDS_optimized: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.ism_lineage_stopping, bool):
            raise ValueError("ism_lineage_stopping must be boolean")
        for name in ("ism_min_sweeps", "ism_patience"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError(f"{name} must be an integer at least 2")
        for name in ("ism_absolute_gain", "ism_relative_gain", "ism_knee_slope_ratio"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.ism_knee_slope_ratio > 1:
            raise ValueError("ism_knee_slope_ratio must not exceed 1")
        if not isinstance(self.is_CDS_optimized, bool):
            raise ValueError("is_CDS_optimized must be a boolean")
        if isinstance(self.parent_pool_multiplier, (bool, np.bool_)) or not isinstance(
            self.parent_pool_multiplier, (int, np.integer)
        ) or self.parent_pool_multiplier < 1:
            raise ValueError("parent_pool_multiplier must be a positive integer")
        if isinstance(self.adalead_mu, (bool, np.bool_)) or not isinstance(self.adalead_mu, (int, float, np.number)) or not np.isfinite(self.adalead_mu) or self.adalead_mu <= 0:
            raise ValueError("adalead_mu must be finite and positive")
        if (isinstance(self.adalead_max_rollout, (bool, np.bool_))
                or not isinstance(self.adalead_max_rollout, (int, np.integer))
                or self.adalead_max_rollout < 1):
            raise ValueError("adalead_max_rollout must be a positive integer")
        if not isinstance(self.adalead_allow_equal, (bool, np.bool_)):
            raise ValueError("adalead_allow_equal must be boolean")
        value = self.adalead_exploration_fraction
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError("adalead_exploration_fraction must lie in [0, 1)")
        value = self.adalead_quality_power
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value <= 0:
            raise ValueError("adalead_quality_power must be finite and positive")
        if self.adalead_lineages is not None and (
            isinstance(self.adalead_lineages, (bool, np.bool_))
            or not isinstance(self.adalead_lineages, (int, np.integer))
            or self.adalead_lineages < 1
        ):
            raise ValueError("adalead_lineages must be a positive integer or None")
        if (isinstance(self.annealing_iterations, (bool, np.bool_))
                or not isinstance(self.annealing_iterations, (int, np.integer))
                or self.annealing_iterations < 1):
            raise ValueError("annealing_iterations must be a positive integer")
        for name in ("annealing_initial_temperature", "annealing_final_temperature"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
                    or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.annealing_final_temperature > self.annealing_initial_temperature:
            raise ValueError(
                "annealing_final_temperature must not exceed annealing_initial_temperature"
            )
        if self.fixed_region is not None:
            if isinstance(self.fixed_region, (str, bytes)) or not isinstance(
                self.fixed_region, Sequence
            ):
                raise ValueError("fixed_region must be a list of positions or slices")
            object.__setattr__(self, "fixed_region", tuple(self.fixed_region))
        if (
            isinstance(self.diversity, bool)
            or not isinstance(self.diversity, (int, np.integer))
            or self.diversity < 0
        ):
            raise ValueError("diversity must be a non-negative integer")
        object.__setattr__(self, "diversity", int(self.diversity))
        if not 0 < self.basesearch_decay < 1:
            raise ValueError("basesearch_decay must be between zero and one")
        for name in (
            "basesearch_queue_size",
            "basesearch_max_stalled",
            "basesearch_batch_size",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
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
        if (
            isinstance(self.cross_entropy_batch_size, bool)
            or not isinstance(self.cross_entropy_batch_size, (int, np.integer))
            or self.cross_entropy_batch_size < 1
        ):
            raise ValueError("cross_entropy_batch_size must be a positive integer")
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
    constraint_scores: np.ndarray | None = None
    constraint_names: tuple[str, ...] = ()

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
        self._validate_constraint_scores()

    def _validate_constraint_scores(self) -> None:
        if self.constraint_scores is None:
            if self.constraint_names:
                raise ValueError("constraint_names require constraint_scores")
            return
        constraint_scores = np.asarray(self.constraint_scores, dtype=np.float64)
        if constraint_scores.ndim == 1:
            constraint_scores = constraint_scores[:, None]
        if constraint_scores.ndim != 2 or len(constraint_scores) != len(self.sequences):
            raise ValueError(
                "constraint_scores must have shape [num_candidates, constraints]"
            )
        if not np.isfinite(constraint_scores).all() or np.any(
            (constraint_scores < 0) | (constraint_scores > 1)
        ):
            raise ValueError("constraint_scores must be finite values in [0, 1]")
        names = tuple(self.constraint_names)
        if len(names) != constraint_scores.shape[1] or len(set(names)) != len(names):
            raise ValueError("constraint_names must identify every constraint column")
        self.constraint_scores = constraint_scores
        self.constraint_names = names

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

    def attach_constraint_scores(
        self,
        constraint_scores: np.ndarray,
        constraint_names: Sequence[str],
    ) -> "OptimizationResult":
        """Attach the pass probabilities returned by each constraint."""
        self.constraint_scores = constraint_scores
        self.constraint_names = tuple(constraint_names)
        self._validate_constraint_scores()
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
        if self.constraint_scores is not None:
            for index, name in enumerate(self.constraint_names):
                frame[f"constraint_{name}"] = self.constraint_scores[:, index]
        if self.activities is not None:
            for index, name in enumerate(self.task_names):
                frame[f"activity_{name}"] = self.activities[:, index]
        evaluation = self.metadata.get("objective_evaluation")
        if evaluation is not None:
            frame["worst_margin"] = evaluation["worst_margin"]
            for index, name in enumerate(evaluation["task_names"]):
                frame[f"margin_{name}"] = np.asarray(evaluation["margins"])[:, index]
                if evaluation["satisfied"] is not None:
                    frame[f"satisfied_{name}"] = np.asarray(evaluation["satisfied"])[
                        :, index
                    ]
            if evaluation["all_satisfied"] is not None:
                frame["all_satisfied"] = evaluation["all_satisfied"]
        return frame
