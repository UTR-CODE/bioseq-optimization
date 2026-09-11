"""User-defined hard and probabilistic sequence constraints."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np

ConstraintOutput: TypeAlias = float | Sequence[float] | np.ndarray
ConstraintFunction: TypeAlias = Callable[
    [str | Sequence[str] | np.ndarray], ConstraintOutput
]
ConstraintSpec: TypeAlias = ConstraintFunction | Sequence[ConstraintFunction]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _validate_probabilities(result: Any, expected_rows: int) -> np.ndarray:
    probabilities = _to_numpy(result)
    if probabilities.ndim == 0:
        probabilities = probabilities.reshape(1)
    if probabilities.ndim != 1 or len(probabilities) != expected_rows:
        raise ValueError("constraint must return one value per sequence with shape [N]")
    if not np.isfinite(probabilities).all():
        raise ValueError("constraint values must be finite")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("constraint values must lie in [0, 1]")
    return probabilities


def evaluate_constraint(
    constraint: ConstraintFunction,
    sequences: Sequence[str] | np.ndarray,
) -> np.ndarray:
    """Evaluate a batch-aware or single-sequence constraint callable."""
    if not callable(constraint):
        raise TypeError("each constraint must be callable")
    values = np.asarray(sequences, dtype=str).reshape(-1).copy()
    try:
        result = constraint(values)
    except (AttributeError, TypeError, ValueError):
        result = [constraint(str(sequence)) for sequence in values]
    else:
        if _to_numpy(result).ndim == 0:
            result = [constraint(str(sequence)) for sequence in values]
    return _validate_probabilities(result, len(values))


def normalize_constraints(
    constraint: ConstraintSpec | None,
) -> tuple[ConstraintFunction, ...]:
    if constraint is None:
        return ()
    functions = (constraint,) if callable(constraint) else tuple(constraint)
    if not functions or any(not callable(function) for function in functions):
        raise TypeError(
            "constraint must be a callable or a nonempty sequence of callables"
        )
    return functions


class ConstraintEvaluator:
    """Evaluate independent constraints once per unique sequence."""

    def __init__(self, constraint: ConstraintSpec, rng: np.random.Generator) -> None:
        self.functions = normalize_constraints(constraint)
        self.rng = rng
        self._probabilities: dict[str, np.ndarray] = {}
        self._passes: dict[str, np.ndarray] = {}
        self.evaluated_counts = np.zeros(len(self.functions), dtype=np.int64)
        self.passed_counts = np.zeros(len(self.functions), dtype=np.int64)
        self.intersection_passed_count = 0
        self.names = self._names()

    def _names(self) -> tuple[str, ...]:
        names: list[str] = []
        used: set[str] = set()
        for index, function in enumerate(self.functions):
            candidate = getattr(function, "__name__", "")
            if not candidate or candidate == "<lambda>":
                candidate = f"constraint_{index}"
            base = candidate
            suffix = 2
            while candidate in used:
                candidate = f"{base}_{suffix}"
                suffix += 1
            names.append(candidate)
            used.add(candidate)
        return tuple(names)

    def evaluate(self, sequences: Sequence[str] | np.ndarray) -> np.ndarray:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        missing = list(
            dict.fromkeys(
                str(sequence)
                for sequence in values
                if str(sequence) not in self._passes
            )
        )
        if missing:
            batch = np.asarray(missing, dtype=str)
            probabilities = np.column_stack(
                [evaluate_constraint(function, batch) for function in self.functions]
            )
            passes = self.rng.random(probabilities.shape) < probabilities
            for sequence, probability, passed in zip(batch, probabilities, passes):
                self._probabilities[str(sequence)] = probability.copy()
                self._passes[str(sequence)] = passed.copy()
            self.evaluated_counts += len(batch)
            self.passed_counts += passes.sum(axis=0)
            self.intersection_passed_count += int(passes.all(axis=1).sum())
        return np.asarray(
            [self._passes[str(sequence)].all() for sequence in values], dtype=bool
        )

    def probabilities_for(self, sequences) -> np.ndarray:
        values = np.asarray(sequences, dtype=str).reshape(-1)
        missing = [
            str(sequence)
            for sequence in values
            if str(sequence) not in self._probabilities
        ]
        if missing:
            raise KeyError(
                f"constraint probabilities were not recorded for {len(missing)} sequences"
            )
        return np.stack([self._probabilities[str(sequence)] for sequence in values])

    def describe(self) -> dict[str, object]:
        evaluated = self.evaluated_counts.astype(float)
        rates = np.divide(
            self.passed_counts,
            evaluated,
            out=np.zeros_like(evaluated),
            where=evaluated > 0,
        )
        total = int(evaluated[0]) if len(evaluated) else 0
        return {
            "mode": "independent_intersection",
            "constraints": [
                {
                    "name": name,
                    "evaluated_count": int(count),
                    "passed_count": int(passed),
                    "pass_rate": float(rate),
                }
                for name, count, passed, rate in zip(
                    self.names, self.evaluated_counts, self.passed_counts, rates
                )
            ],
            "intersection_passed_count": self.intersection_passed_count,
            "intersection_pass_rate": (
                self.intersection_passed_count / total if total else 0.0
            ),
        }
