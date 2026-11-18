"""Dataset-independent discrete sequence optimization algorithms."""

from __future__ import annotations

import inspect
import logging
from dataclasses import replace
import random
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

try:  # Optional compiled CPU fast path; NumPy/Python remains the fallback.
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised in minimal installations
    njit = None
    prange = range

from .api import (
    _sequence_space,
    CachedScorer,
    FixedRegionSequenceSpace,
    OptimizationConfig,
    OptimizationResult,
    ScoreBatch,
    SequenceScorer,
    SequenceSpace,
    normalize_nucleotide_sequences,
    resolve_fixed_positions,
)
from .constraints import ConstraintSpec
from .objectives import MultiTaskObjective, ObjectiveScorer, TargetObjective
from .saturation import SaturationConfig, SaturationResult

Optimizer = Callable[
    [np.ndarray, CachedScorer, SequenceSpace, OptimizationConfig, np.random.Generator],
    OptimizationResult,
]
_REGISTRY: dict[str, Optimizer] = {}
_DEFAULT_BUDGET = 10_000


def register_algorithm(name: str):
    """Register an optimizer function under a stable user-facing name."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("algorithm name must not be empty")

    def decorator(function: Optimizer) -> Optimizer:
        if key in _REGISTRY:
            raise ValueError(f"algorithm already registered: {key}")
        _REGISTRY[key] = function
        return function

    return decorator


def available_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _record(
    archive: dict[str, tuple[float, float]], sequences: np.ndarray, batch: ScoreBatch
) -> None:
    uncertainty = (
        np.zeros(len(batch.scores))
        if batch.uncertainties is None
        else batch.uncertainties
    )
    for sequence, score, std in zip(sequences, batch.scores, uncertainty):
        key = str(sequence)
        if key not in archive or float(score) > archive[key][0]:
            archive[key] = (float(score), float(std))


def _score(
    scorer: CachedScorer, archive: dict[str, tuple[float, float]], sequences: np.ndarray
) -> ScoreBatch:
    batch = scorer.score(sequences)
    passed = scorer.constraints_passed(sequences)
    if passed.any():
        _record(
            archive,
            np.asarray(sequences)[passed],
            ScoreBatch(
                batch.scores[passed],
                None if batch.uncertainties is None else batch.uncertainties[passed],
            ),
        )
    return batch


_DNA_CODES = np.full(256, 255, dtype=np.uint8)
_DNA_CODES[np.frombuffer(b"A", dtype=np.uint8)] = 0
_DNA_CODES[np.frombuffer(b"C", dtype=np.uint8)] = 1
_DNA_CODES[np.frombuffer(b"G", dtype=np.uint8)] = 2
_DNA_CODES[np.frombuffer(b"T", dtype=np.uint8)] = 3
# For an XOR of two 2-bit DNA symbols, either bit being set means mismatch.
_PACKED_MISMATCHES = np.asarray(
    [sum(bool((value >> shift) & 3) for shift in (0, 2, 4, 6)) for value in range(256)],
    dtype=np.uint8,
)


if njit is not None:

    @njit(cache=True, nogil=True)
    def _greedy_indices_numba(codes, count, diversity, start):
        """Sequential greedy selection in compiled native code."""
        selected = np.full(count, -1, dtype=np.int64)
        selected[0] = start
        selected_count = 1
        for index in range(codes.shape[0]):
            if index == start:
                continue
            accepted = True
            for selected_offset in range(selected_count):
                previous = selected[selected_offset]
                mismatches = 0
                for position in range(codes.shape[1]):
                    if codes[index, position] != codes[previous, position]:
                        mismatches += 1
                        if mismatches >= diversity:
                            break
                if mismatches < diversity:
                    accepted = False
                    break
            if accepted:
                selected[selected_count] = index
                selected_count += 1
                if selected_count == count:
                    break
        return selected, selected_count

    @njit(cache=True, nogil=True, parallel=True)
    def _alternative_indices_numba(codes, count, diversity, starts):
        """Evaluate independent fallback starts across available CPU cores."""
        selected = np.full((starts, count), -1, dtype=np.int64)
        lengths = np.zeros(starts, dtype=np.int64)
        for offset in prange(starts):
            start = offset + 1
            selected[offset, 0] = start
            selected_count = 1
            for index in range(codes.shape[0]):
                if index == start:
                    continue
                accepted = True
                for selected_offset in range(selected_count):
                    previous = selected[offset, selected_offset]
                    mismatches = 0
                    for position in range(codes.shape[1]):
                        if codes[index, position] != codes[previous, position]:
                            mismatches += 1
                            if mismatches >= diversity:
                                break
                    if mismatches < diversity:
                        accepted = False
                        break
                if accepted:
                    selected[offset, selected_count] = index
                    selected_count += 1
                    if selected_count == count:
                        break
            lengths[offset] = selected_count
        return selected, lengths

else:
    _greedy_indices_numba = None
    _alternative_indices_numba = None


def _encode_packed(sequences: Sequence[str]) -> tuple[np.ndarray, int]:
    """Encode DNA strings as four 2-bit bases per byte."""
    values = np.frombuffer("".join(str(sequence) for sequence in sequences).encode("ascii"), dtype=np.uint8)
    length = len(sequences[0]) if sequences else 0
    codes = _DNA_CODES[values].reshape(len(sequences), length)
    packed = np.zeros((len(sequences), (length + 3) // 4), dtype=np.uint8)
    # Pack each offset into its corresponding 2-bit lane.
    for offset in range(4):
        columns = codes[:, offset::4]
        packed[:, : columns.shape[1]] |= columns << np.uint8(2 * offset)
    return packed, length


def _hamming_distance(left: str, right: str) -> int:
    """Exact scalar Hamming distance, retained for public/internal callers."""
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length sequences")
    left_bits = _hamming_bits(str(left))
    right_bits = _hamming_bits(str(right))
    mask = int.from_bytes(b"\x01" * len(left), "little")
    xor = left_bits ^ right_bits
    xor |= xor >> 4
    xor |= xor >> 2
    xor |= xor >> 1
    return (xor & mask).bit_count()


def _packed_distance(packed: np.ndarray, index: int, other: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(other, packed[index])
    return _PACKED_MISMATCHES[xor].sum(axis=1, dtype=np.uint16)


def _distance_matrix(sequences: Sequence[str]) -> tuple[np.ndarray, int]:
    """Compute exact all-pairs distances using benchmark-selected backends."""
    packed, length = _encode_packed(sequences)
    n = len(sequences)
    # GPU matmul wins for resident medium-sized matrices. This path is bounded
    # to avoid allocating quadratic tensors for large saturation archives.
    if n >= 512 and n <= 4096 and n * length >= 250_000:
        try:
            import torch

            if torch.cuda.is_available():
                codes = _DNA_CODES[np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)].reshape(n, length)
                onehot = torch.nn.functional.one_hot(torch.from_numpy(codes.astype(np.int64)), 4).reshape(n, length * 4).to(device="cuda", dtype=torch.float32)
                distances = (length - onehot @ onehot.T).to(torch.int32).cpu().numpy().astype(np.uint16)
                return distances, length
        except (ImportError, RuntimeError):
            pass
    distances = np.empty((n, n), dtype=np.uint16)
    for index in range(n):
        distances[index] = _packed_distance(packed, index, packed)
    return distances, length


def _rank_archive(
    archive: dict[str, tuple[float, float]],
    scorer: CachedScorer,
) -> list[tuple[str, tuple[float, float]]]:
    sequences = np.asarray(list(archive), dtype=str)
    scores = scorer.current_scores(sequences)
    uncertainties = scorer.current_uncertainties(sequences)
    ranked = [
        (sequence, (float(score), float(uncertainty)))
        for sequence, score, uncertainty in zip(sequences, scores, uncertainties)
    ]
    return sorted(ranked, key=lambda item: (-item[1][0], item[0]))


@lru_cache(maxsize=131072)
def _hamming_bits(sequence: str) -> int:
    # Python's arbitrary-width XOR and bit_count run in compiled code. Byte
    # lanes preserve every ASCII symbol (including non-DNA custom alphabets).
    return int.from_bytes(sequence.encode("ascii"), "little")


def _hamming_at_least(left: str, right: str, minimum: int) -> bool:
    """Return whether two ASCII sequences meet a Hamming threshold."""
    if minimum <= 0:
        return True
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length sequences")
    xor = _hamming_bits(str(left)) ^ _hamming_bits(str(right))
    xor |= xor >> 4
    xor |= xor >> 2
    xor |= xor >> 1
    mask = int.from_bytes(b"\x01" * len(left), "little")
    return (xor & mask).bit_count() >= minimum


def _select_diverse(
    ranked: Sequence[tuple[str, tuple[float, float]]],
    count: int,
    diversity: int,
    *,
    fallback: bool = True,
) -> list[tuple[str, tuple[float, float]]]:
    if count == 1 or diversity == 0:
        return list(ranked[:count])
    if not ranked:
        return []
    length = len(ranked[0][0])

    # Hamming checks dominate long-sequence saturation runs.  Native loops
    # retain early exits (important for closely related candidates), avoid
    # quadratic distance matrices, and release the GIL.  Fallback starting
    # points are independent and therefore evaluated across CPU cores.
    if (
        _greedy_indices_numba is not None
        and len(ranked) * length >= 10_000
        and all(len(item[0]) == length for item in ranked)
    ):
        encoded = np.frombuffer(
            "".join(str(item[0]) for item in ranked).encode("ascii"),
            dtype=np.uint8,
        ).reshape(len(ranked), length)
        indices, selected_count = _greedy_indices_numba(
            encoded, count, diversity, 0
        )
        selected = [ranked[int(index)] for index in indices[:selected_count]]
        if selected_count == count or not fallback:
            return selected
        starts = min(len(ranked), 32) - 1
        if starts <= 0:
            return selected
        alternatives, lengths = _alternative_indices_numba(
            encoded, count, diversity, starts
        )
        complete = np.flatnonzero(lengths == count)
        if len(complete):
            offset = int(complete[0])
            return [ranked[int(index)] for index in alternatives[offset, :count]]
        best_offset = int(np.argmax(lengths))
        if int(lengths[best_offset]) > len(selected):
            return [
                ranked[int(index)]
                for index in alternatives[best_offset, : lengths[best_offset]]
            ]
        return selected

    mask = int.from_bytes(b"\x01" * length, "little")

    def select(start):
        selected = [ranked[start]]
        bits = [_hamming_bits(str(ranked[start][0]))]
        for index, item in enumerate(ranked):
            if index == start:
                continue
            value = _hamming_bits(str(item[0]))
            for previous in bits:
                xor = value ^ previous
                xor |= xor >> 4
                xor |= xor >> 2
                xor |= xor >> 1
                if (xor & mask).bit_count() < diversity:
                    break
            else:
                selected.append(item)
                bits.append(value)
                if len(selected) == count:
                    break
        return selected

    selected = select(0)
    if len(selected) == count or not fallback:
        return selected
    best = selected
    for start in range(1, min(len(ranked), 32)):
        alternative = select(start)
        if len(alternative) == count:
            return alternative
        if len(alternative) > len(best):
            best = alternative
    return best


def _finalize(
    name: str,
    archive: dict[str, tuple[float, float]],
    config: OptimizationConfig,
    scorer: CachedScorer,
    metadata: dict[str, Any],
    trajectory: list[dict[str, float | int]],
) -> OptimizationResult:
    if len(archive) < config.num_candidates:
        raise RuntimeError(
            f"only {len(archive)} unique sequences were scored; {config.num_candidates} required; "
            "increase the initial population or search budget"
        )
    ranked = _select_diverse(
        _rank_archive(archive, scorer),
        config.num_candidates,
        config.diversity,
    )
    if len(ranked) < config.num_candidates:
        raise RuntimeError(
            f"only {len(ranked)} sequences satisfy diversity={config.diversity}; "
            "increase the initial population or budget, or lower diversity"
        )
    metadata = {
        **metadata,
        "diversity": config.diversity,
        "stop_reason": (
            "target_reached"
            if bool(getattr(scorer.scorer, "target_stop_reached", False))
            else metadata.get("stop_reason", "budget")
        ),
    }
    return OptimizationResult(
        name,
        np.asarray([item[0] for item in ranked], dtype=str),
        np.asarray([item[1][0] for item in ranked], dtype=np.float64),
        np.asarray([item[1][1] for item in ranked], dtype=np.float64),
        scorer.query_count,
        metadata,
        trajectory,
    )


def _initial(
    initial,
    space: SequenceSpace,
    config: OptimizationConfig,
    rng: np.random.Generator,
    *,
    anchor: str | None = None,
) -> np.ndarray:
    maximum_distance = int(getattr(space, "max_hamming_distance", space.length))
    capacity = getattr(space, "capacity", space.alphabet_size**space.length)
    if config.num_candidates > capacity:
        raise ValueError(
            f"requested {config.num_candidates} unique candidates exceeds sequence space capacity {capacity}"
        )
    if config.num_candidates > 1 and config.diversity > maximum_distance:
        raise ValueError(
            f"diversity={config.diversity} exceeds the {maximum_distance} mutable "
            "positions available for pairwise differences; "
            "lower diversity or request one candidate"
        )
    if initial is None:
        capacity = getattr(space, "capacity", space.alphabet_size**space.length)
        if config.num_candidates > capacity:
            raise ValueError(
                f"cannot draw {config.num_candidates} unique seeds from a space of size {capacity}"
            )
        seeds: dict[str, None] = {}
        if anchor is not None:
            anchor_value = str(space.validate([anchor])[0])
            seeds[anchor_value] = None
        draws = 0
        maximum_draws = max(10_000, 1000 * config.num_candidates)
        while len(seeds) < config.num_candidates and draws < maximum_draws:
            needed = config.num_candidates - len(seeds)
            candidates = space.random(min(max(64, needed), 4096), rng)
            draws += len(candidates)
            for sequence in candidates:
                value = str(sequence)
                if config.num_candidates == 1 or all(
                    _hamming_distance(value, chosen) >= config.diversity
                    for chosen in seeds
                ):
                    seeds.setdefault(value, None)
                if len(seeds) == config.num_candidates:
                    break
        if len(seeds) < config.num_candidates:
            raise ValueError(
                f"cannot draw {config.num_candidates} seeds satisfying "
                f"diversity={config.diversity}; lower diversity or supply initial"
            )
        values = np.asarray(list(seeds), dtype=str)
    else:
        values = space.validate(initial)
    if not len(values):
        raise ValueError("initial must contain at least one sequence")
    if len(values) < config.num_candidates:
        raise ValueError(
            f"initial must contain at least {config.num_candidates} sequences; "
            f"received {len(values)}"
        )
    if len(values) > config.budget:
        raise ValueError(
            f"initial contains {len(values)} sequences but budget is {config.budget}"
        )
    if len(np.unique(values)) != len(values):
        raise ValueError("initial sequences must be unique")
    return values.copy()


def _constraint_initial(
    initial: np.ndarray,
    scorer: CachedScorer,
    space: SequenceSpace,
    config: OptimizationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if scorer.constraint_evaluator is None:
        return initial
    passed = scorer.hard_constraints_passed(initial)
    target_count = len(initial)
    accepted = [str(sequence) for sequence, keep in zip(initial, passed) if keep]
    seen = {str(sequence) for sequence in initial}
    capacity = getattr(space, "capacity", space.alphabet_size**space.length)
    maximum_unique = min(capacity, max(10_000, 1000 * target_count))
    draws = 0
    maximum_draws = max(10_000, 10 * maximum_unique)
    while (
        len(accepted) < target_count
        and len(seen) < maximum_unique
        and draws < maximum_draws
    ):
        count = min(max(64, target_count - len(accepted)), 4096)
        candidates = space.random(count, rng)
        draws += len(candidates)
        novel = np.asarray(
            list(
                dict.fromkeys(
                    str(sequence)
                    for sequence in candidates
                    if str(sequence) not in seen
                )
            ),
            dtype=str,
        )
        seen.update(map(str, novel))
        if not len(novel):
            continue
        keep = scorer.hard_constraints_passed(novel)
        accepted.extend(str(sequence) for sequence, value in zip(novel, keep) if value)
    if len(accepted) < target_count:
        raise ValueError(
            "constraints did not admit enough unique initial sequences; "
            "supply less restrictive constraints or feasible initial sequences"
        )
    return np.asarray(accepted[:target_count], dtype=str)


def _prediction_initial(initial, scorer, space, config, rng):
    """Build a feasible seed population, charging all prediction queries.

    Bypass saturation tracking during this initialization-only screen. Search
    sees only feasible seeds, and its first scored batch establishes baseline.
    Never fill missing slots with infeasible sequences.
    """
    if not getattr(scorer.scorer, "has_prediction_bounds", False):
        return initial
    count = len(initial)
    accepted = {}
    seen = set()
    proposals = initial
    max_draws = max(10000, 1000 * count)
    draws = 0
    # Reserve one batch for the optimizer's initial score call (including
    # optimizers that count repeated proposals).
    while len(accepted) < count and draws < max_draws:
        room = max_draws - draws
        if scorer.remaining is not None:
            room = min(room, scorer.remaining - count)
        if room <= 0:
            break
        proposals = np.asarray([s for s in proposals if str(s) not in seen], dtype=str)[:room]
        if len(proposals):
            seen.update(map(str, proposals))
            CachedScorer.score(scorer, proposals)
            keep = scorer.constraints_passed(proposals)
            accepted.update((str(s), None) for s in proposals[keep])
        if len(accepted) >= count:
            break
        size = min(max(64, count - len(accepted)), room, max_draws - draws)
        proposals = space.random(size, rng)
        draws += size
    if len(accepted) < count:
        raise ValueError(
            f"stop_values admitted only {len(accepted)} feasible initial sequences; "
            f"{count} required. Supply feasible initial sequences, increase the "
            "budget, or relax prediction bounds. No out-of-bounds seeds retained."
        )
    return np.asarray(list(accepted)[:count], dtype=str)


@register_algorithm("random")
def random_search(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    _score(scorer, archive, initial)
    while scorer.remaining is None or scorer.remaining > 0:
        count = min(4096, scorer.remaining or 4096)
        before = scorer.query_count
        _score(scorer, archive, space.random(count, rng))
        if scorer.query_count == before:
            break
    return _finalize(
        "random", archive, config, scorer, {"strategy": "uniform random screening"}, []
    )


def _ism_curve_stop(history, config):
    """Online stopping: confirm a low slope or a knee with a sustained slow tail.

    This uses only observed gains; it does not claim a global optimum or a
    fixed fraction of future achievable gain. A linear curve has no knee.
    """
    y = np.asarray(history, dtype=float)
    rounds = len(y) - 1
    window = config.ism_patience
    if rounds < max(config.ism_min_sweeps, window + 2):
        return None, None
    gain = max(0.0, float(y[-1] - y[0]))
    tail_slope = float(y[-1] - y[-1-window]) / window
    floor = max(config.ism_absolute_gain, config.ism_relative_gain * gain)
    if tail_slope <= floor:
        return "small_gain", None
    if gain <= 0:
        return None, None
    x = np.arange(len(y), dtype=float) / rounds
    distance = (y - y[0]) / gain - x
    knee = int(np.argmax(distance))
    if knee >= 2 and rounds - knee >= window and distance[knee] >= 0.05:
        head_slope = float(y[knee] - y[0]) / knee
        post_slope = float(y[-1] - y[knee]) / (rounds - knee)
        if max(tail_slope, post_slope) <= config.ism_knee_slope_ratio * head_slope:
            return "knee_slow_tail", knee
    return None, None


@register_algorithm("ism_greedy")
def ism_greedy(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    current = initial.copy()
    current_scores = _score(scorer, archive, current).scores
    lineages = [dict(lineage=i, initial_sequence=str(s), scores=[float(v)],
                     stop_reason=None, stop_sweep=None, knee_sweep=None)
                for i, (s, v) in enumerate(zip(current, current_scores))]
    ids = np.arange(len(current))
    active = np.ones(len(current), dtype=bool)
    objective_version = scorer.objective_version
    diagnostics = dict(lineages=lineages, rounds=[], objective_resets=0,
                       stopping_enabled=config.ism_lineage_stopping,
                       minimum_sweeps=config.ism_min_sweeps, patience=config.ism_patience,
                       absolute_gain=config.ism_absolute_gain, relative_gain=config.ism_relative_gain,
                       knee_slope_ratio=config.ism_knee_slope_ratio)
    scorer.ism_diagnostics = diagnostics  # Also survives global plateau exceptions.
    trajectory = [dict(query_count=scorer.query_count, best_score=float(current_scores.max()),
                       mean_score=float(current_scores.mean()), active_lineages=int(active.sum()))]
    mutable_count = len(getattr(space, "mutable_positions", tuple(range(space.length))))
    block = int(getattr(space, "single_mutant_count", mutable_count * (space.alphabet_size - 1)))
    if block == 0:
        return _finalize("ism_greedy", archive, config, scorer,
                         {"sweeps": 0, "stop_reason": "fixed_region_exhausted"}, trajectory)
    sweeps = 0
    while (scorer.remaining is None or scorer.remaining >= block) and active.any():
        before = scorer.query_count
        # Preserve seeded ordering over all lineages, then omit frozen branches.
        order = rng.permutation(len(current))
        order = order[active[order]]
        count = len(order) if scorer.remaining is None else min(len(order), scorer.remaining // block)
        chosen = order[:count]
        if not len(chosen):
            break
        mutants = space.single_mutants(current[chosen])
        # Complete the lineage updates before notifying global plateau tracking.
        grouped = hasattr(scorer, "begin_observation_group")
        if grouped:
            scorer.begin_observation_group()
        batch = _score(scorer, archive, mutants)
        passed = scorer.constraints_passed(mutants)
        current_scores = scorer.current_scores(current)
        if scorer.objective_version != objective_version:
            objective_version = scorer.objective_version
            active[:] = True
            diagnostics['objective_resets'] += 1
            for index, lineage in enumerate(ids):
                item = lineages[int(lineage)]
                item.update(scores=[float(current_scores[index])], stop_reason=None,
                            stop_sweep=None, knee_sweep=None)
        sweeps += 1
        for offset, index in enumerate(chosen):
            values = batch.scores[offset * block:(offset + 1) * block]
            valid = passed[offset * block:(offset + 1) * block]
            old = float(current_scores[index])
            if valid.any():
                best = int(np.flatnonzero(valid)[np.argmax(values[valid])])
                if values[best] > old:
                    current[index] = mutants[offset * block + best]
                    current_scores[index] = values[best]
            item = lineages[int(ids[index])]
            item['scores'].append(float(current_scores[index]))
            if config.ism_lineage_stopping:
                reason, knee = _ism_curve_stop(item['scores'], config)
                # An exhaustive neighborhood with no improvement is terminal too.
                if current_scores[index] <= old:
                    reason = "no_improving_neighbor"
                if reason:
                    active[index] = False
                    item.update(stop_reason=reason, stop_sweep=sweeps, knee_sweep=knee)
        retained = scorer.constraints_passed(current)
        for lineage in ids[~retained]:
            lineages[int(lineage)].update(stop_reason='constraint_rejected', stop_sweep=sweeps)
        current, active, ids = current[retained], active[retained], ids[retained]
        current_scores = scorer.current_scores(current)
        diagnostics['rounds'].append(dict(sweep=sweeps, scanned_lineages=len(chosen),
            active_lineages=int(active.sum()), proposals=len(mutants), query_count=scorer.query_count))
        trajectory.append(dict(query_count=scorer.query_count,
            best_score=float(current_scores.max()) if len(current_scores) else float('-inf'),
            mean_score=float(current_scores.mean()) if len(current_scores) else float('-inf'),
            active_lineages=int(active.sum())))
        logging.getLogger(__name__).info(
            'ISM sweep=%d scanned=%d active=%d mean=%.6f proposals=%d',
            sweeps, len(chosen), int(active.sum()),
            float(current_scores.mean()) if len(current_scores) else float('nan'), len(mutants))
        if grouped:
            scorer.end_observation_group()
        if scorer.query_count == before:
            break
    return _finalize("ism_greedy", archive, config, scorer,
        {"sweeps": sweeps, "stop_reason": "lineage_saturation" if not active.any() else "algorithm_stopped",
         "ism_lineage_diagnostics": diagnostics}, trajectory)


@register_algorithm("simulated_annealing")
def simulated_annealing(initial, scorer, space, config, rng):
    # Repeated stochastic proposals still advance patience/budget; the model
    # cache continues to avoid duplicate inference.
    scorer.count_repeated = True
    archive: dict[str, tuple[float, float]] = {}
    current = initial.copy()
    current_scores = _score(scorer, archive, current).scores
    trajectory = []
    iterations = 0
    while (
        (iterations < config.annealing_iterations or scorer.remaining is None)
        and (scorer.remaining is None or scorer.remaining >= len(current))
    ):
        progress = (
            1.0
            if config.annealing_iterations == 1
            else min(1.0, iterations / (config.annealing_iterations - 1))
        )
        temp = (
            config.annealing_initial_temperature
            + progress
            * (config.annealing_final_temperature - config.annealing_initial_temperature)
        )
        proposals = space.mutate_one(current, rng)
        values = _score(scorer, archive, proposals).scores
        passed = scorer.constraints_passed(proposals)
        current_scores = scorer.current_scores(current)
        current_passed = scorer.constraints_passed(current)
        accept = np.zeros(len(current), dtype=bool)
        accept[passed] = (
            (~current_passed[passed])
            | (values[passed] >= current_scores[passed])
            | (
                rng.random(int(passed.sum()))
                < np.exp(
                    np.minimum((values[passed] - current_scores[passed]) / temp, 0)
                )
            )
        )
        current[accept] = proposals[accept]
        current_scores[accept] = values[accept]
        retained = scorer.constraints_passed(current)
        current = current[retained]
        current_scores = scorer.current_scores(current)
        iterations += 1
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "iteration": iterations,
                "temperature": float(temp),
                "best_score": float(current_scores.max()),
                "mean_score": float(current_scores.mean()),
            }
        )
    return _finalize(
        "simulated_annealing",
        archive,
        config,
        scorer,
        {
            "iterations": iterations,
            "maximum_iterations": config.annealing_iterations,
            "initial_temperature": config.annealing_initial_temperature,
            "final_temperature": config.annealing_final_temperature,
            "last_temperature": (
                None if not trajectory else trajectory[-1]["temperature"]
            ),
            "temperature_schedule": "linear_by_iteration",
        },
        trajectory,
    )


@register_algorithm("genetic_algorithm")
def genetic_algorithm(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    population = initial.copy()
    scores = _score(scorer, archive, population).scores
    population_size = len(population)
    elite_count = max(1, int(round(population_size * config.elite_fraction)))
    trajectory = []
    while scorer.remaining is None or scorer.remaining >= population_size:
        before = scorer.query_count
        retained = scorer.constraints_passed(population)
        population = population[retained]
        if not len(population):
            break
        scores = scorer.current_scores(population)
        parent_count = min(elite_count, len(population))
        order = np.lexsort((population, -scores))
        pool = order[:min(len(population), config.parent_pool_multiplier * parent_count)]
        elite, _ = _diverse_parent_subset(population[pool], scores[pool], parent_count)
        if hasattr(space, "offspring"):
            children = space.offspring(
                elite, population_size, config.mutation_rate, rng
            )
        else:
            a = space.encode(elite[rng.integers(len(elite), size=population_size)])
            b = space.encode(elite[rng.integers(len(elite), size=population_size)])
            child = np.where(rng.random(a.shape) < 0.5, a, b)
            mask = rng.random(child.shape) < config.mutation_rate
            child = np.where(
                mask, rng.integers(space.alphabet_size, size=child.shape), child
            )
            children = space.decode(child)
        child_scores = _score(scorer, archive, children).scores
        passed = scorer.constraints_passed(children)
        retained = scorer.constraints_passed(population)
        population = population[retained]
        scores = scorer.current_scores(population)
        combined = np.concatenate([population, children[passed]])
        combined_scores = np.concatenate([scores, child_scores[passed]])
        keep = np.argsort(combined_scores)[-population_size:]
        population, scores = combined[keep], combined_scores[keep]
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(scores.max()),
                "mean_score": float(scores.mean()),
            }
        )
        if scorer.query_count == before:
            break
    return _finalize(
        "genetic_algorithm",
        archive,
        config,
        scorer,
        {"generations": len(trajectory), "elite_count": elite_count,
         "parent_pool_multiplier": config.parent_pool_multiplier,
         "parent_selection": "greedy_max_min_hamming"},
        trajectory,
    )


def _diverse_parent_subset(sequences, scores, count):
    """Score-ranked pool to max-min parents; retain the best as an anchor.

    This is a deterministic greedy approximation, not a global optimum.
    Equal distances favor higher score, then lexicographic sequence order.
    """
    sequences = np.asarray(sequences, dtype=str)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((sequences, -scores))
    sequences, scores = sequences[order], scores[order]
    if len(sequences) <= count:
        return sequences, scores
    codes = np.frombuffer(''.join(sequences).encode('ascii'), dtype=np.uint8).reshape(len(sequences), -1)
    selected = [0]
    distances = np.count_nonzero(codes != codes[0], axis=1)
    distances[0] = -1
    while len(selected) < count:
        index = int(np.argmax(distances))
        selected.append(index)
        distances = np.minimum(distances, np.count_nonzero(codes != codes[index], axis=1))
        distances[selected] = -1
    return sequences[selected], scores[selected]


def _cluster_parent_subset(sequences, scores, count, *, offset=0, cap=10, return_groups=False):
    """Complete-link clusters (20% Hamming diameter), capped round-robin.

    Score-ordered deterministic clustering; every pair in a cluster is within
    the threshold. The rotation advances across calls, with no near-duplicate
    padding if the capped set is smaller than the requested population.
    """
    sequences = np.asarray(sequences, dtype=str)
    scores = np.asarray(scores, dtype=float)
    order = np.lexsort((sequences, -scores))
    unique = list(dict.fromkeys(sequences[order]))
    lookup = {str(sequences[i]): float(scores[i]) for i in order[::-1]}
    if not unique:
        return np.array([], dtype=str), np.array([], dtype=float)
    codes = np.array([list(s) for s in unique], dtype="U1").view(np.uint32)
    threshold = int(len(unique[0]) * .2)
    clusters = []
    compatible = np.ones((len(unique), len(unique)), dtype=bool)
    for start in range(0, len(unique), 64):
        compatible[start:start + 64] = np.count_nonzero(
            codes[start:start + 64, None, :] != codes[None, :, :], axis=2) <= threshold
    for index in range(len(unique)):
        for cluster in clusters:
            if all(compatible[index, member] for member in cluster):
                cluster.append(index)
                break
        else:
            clusters.append([index])
    shift = offset % len(clusters)
    clusters = clusters[shift:] + clusters[:shift]
    selected = [cluster[level] for level in range(cap)
                for cluster in clusters if len(cluster) > level][:count]
    values = np.asarray([unique[i] for i in selected], dtype=str)
    if return_groups:
        retained = set(selected)
        return [np.asarray([unique[i] for i in cluster[:cap] if i in retained], dtype=str)
                for cluster in clusters if any(i in retained for i in cluster[:cap])]
    return values, np.asarray([lookup[s] for s in values])


def _quality_exploration_subset(
    sequences, scores, count, *, rng, exploration_fraction=0.2,
    quality_power=4.0, quality_pool_multiplier=5,
):
    """Keep one best anchor, then sample quality and sequence novelty.

    Quality uses tied empirical percentile ranks, so positive affine score
    transformations leave the sampling distribution unchanged. Exploration
    draws from all remaining feasible candidates, with a soft Hamming-novelty
    preference and no family quota or exclusion radius.
    """
    sequences = np.asarray(sequences, dtype=str)
    scores = np.asarray(scores, dtype=float)
    if len(sequences) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("selection requires aligned finite scores")
    # Canonical identity order makes tied scores/input permutations reproducible.
    lookup = {}
    for sequence, score in zip(sequences, scores):
        key = str(sequence)
        lookup[key] = max(lookup.get(key, -np.inf), float(score))
    values = np.asarray(sorted(lookup), dtype=str)
    fitness = np.asarray([lookup[s] for s in values])
    count = min(int(count), len(values))
    if count <= 0:
        return values[:0], fitness[:0]
    best = int(np.argmax(fitness))
    selected = [best]
    remaining = np.ones(len(values), dtype=bool)
    remaining[best] = False
    exploration_count = (min(count-1, max(1, int(np.floor(count*exploration_fraction))))
                         if exploration_fraction > 0 and count > 1 else 0)
    quality_count = count - exploration_count - 1
    ranks = np.searchsorted(np.sort(fitness), fitness, side='right') / len(fitness)
    # Limit only the quality pool; exploration can reach the entire population.
    pool_size = min(len(values), max(count, int(quality_pool_multiplier*count)))
    cutoff = np.partition(fitness, len(fitness)-pool_size)[len(fitness)-pool_size]
    quality_mask = (fitness >= cutoff) & remaining
    if quality_count:
        eligible = np.flatnonzero(quality_mask)
        weights = np.maximum(ranks[eligible] ** quality_power, 1e-12)
        chosen = rng.choice(eligible, size=quality_count, replace=False, p=weights/weights.sum())
        selected.extend(chosen.tolist())
        remaining[chosen] = False
    if exploration_count:
        codes = np.asarray([list(s) for s in values])
        distances = np.full(len(values), codes.shape[1], dtype=int)
        for index in selected:
            distances = np.minimum(distances, np.count_nonzero(codes != codes[index], axis=1))
        for _ in range(exploration_count):
            eligible = np.flatnonzero(remaining)
            weights = (0.05 + distances[eligible]/max(codes.shape[1], 1)) ** 2
            index = int(rng.choice(eligible, p=weights/weights.sum()))
            selected.append(index)
            remaining[index] = False
            distances = np.minimum(distances, np.count_nonzero(codes != codes[index], axis=1))
    return values[selected], fitness[selected]


def _mutate_expected(sequences, space, expected_mutations, rng):
    """Poisson number of space-native mutation operations (mean mu).

    Unlike FLEXS per-position mutation, this preserves custom, fixed-region
    and synonymous-codon spaces. Zero operations are possible; the rollout's
    bounded novelty retry rejects unchanged or previously evaluated proposals.
    """
    values = np.asarray(sequences, dtype=str).copy()
    counts = rng.poisson(expected_mutations, len(values))
    for step in range(int(counts.max()) if len(counts) else 0):
        mask = counts > step
        values[mask] = space.mutate_one(values[mask], rng)
    return values


@register_algorithm("adalead")
def adalead(initial, scorer, space, config, rng):
    archive: dict[str, tuple[float, float]] = {}
    population = initial.copy()
    scores = _score(scorer, archive, population).scores
    trajectory = []
    accepted_total = 0
    rollout_steps = 0
    seen = set(map(str, initial))
    stop_reason = "budget"
    while scorer.remaining is None or scorer.remaining > 0:
        before = scorer.query_count
        retained = scorer.constraints_passed(population)
        population = population[retained]
        if not len(population):
            break
        scores = scorer.current_scores(population)
        elite_count = max(1, int(round(len(population) * config.elite_fraction)))
        elite, _ = _quality_exploration_subset(
            population, scores, elite_count, rng=rng,
            exploration_fraction=config.adalead_exploration_fraction,
            quality_power=config.adalead_quality_power,
            quality_pool_multiplier=config.parent_pool_multiplier,
        )
        begin_group = getattr(scorer, "begin_observation_group", None)
        end_group = getattr(scorer, "end_observation_group", None)
        if begin_group is not None:
            begin_group()
        proposals_list = []
        # Each elite parent gets an accepted-child rollout. The root score is
        # fixed for the lineage, matching FLEXS and allowing neutral steps.
        current = np.resize(elite, config.adalead_lineages or len(population))
        roots = current.copy()
        for depth in range(config.adalead_max_rollout):
            if not len(current) or scorer.remaining == 0:
                break
            # Batch all active lineages; cap by remaining budget even if
            # repeated proposals count. Bounded retries terminate tiny spaces.
            limit = len(current) if scorer.remaining is None else min(len(current), scorer.remaining)
            pending = list(range(len(current)))
            proposed, indices = [], []
            for attempt in range(32):
                if not pending or len(proposed) >= limit:
                    break
                draws = _mutate_expected(current[pending], space, config.adalead_mu, rng)
                retry = []
                for idx, child in zip(pending, draws):
                    if str(child) in seen:
                        retry.append(idx)
                    else:
                        seen.add(str(child))
                        proposed.append(str(child))
                        indices.append(idx)
                        if len(proposed) == limit:
                            break
                pending = retry
            if not proposed:
                break
            proposal = np.asarray(proposed, dtype=str)
            values = _score(scorer, archive, proposal).scores
            root_scores = scorer.current_scores(roots[indices])
            accepted = scorer.constraints_passed(proposal) & (
                values >= root_scores if config.adalead_allow_equal else values > root_scores
            )
            proposals_list.extend(proposal[accepted].tolist())
            accepted_total += int(accepted.sum())
            rollout_steps += 1
            current = proposal[accepted]
            roots = roots[indices][accepted]
        proposals = np.asarray(proposals_list, dtype=str)
        population_size = len(population)
        retained = scorer.constraints_passed(population)
        candidates = np.concatenate([population[retained], proposals]) if len(proposals) else population[retained]
        if not len(candidates):
            break
        # Use the same quality/exploration mixture for survivors, preserving
        # the best while avoiding hard family quotas or deterministic truncation.
        population, scores = _quality_exploration_subset(
            candidates, scorer.current_scores(candidates), population_size, rng=rng,
            exploration_fraction=config.adalead_exploration_fraction,
            quality_power=config.adalead_quality_power,
            quality_pool_multiplier=config.parent_pool_multiplier,
        )
        if end_group is not None:
            end_group()
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(scores.max()),
                "mean_score": float(scores.mean()),
            }
        )
        if scorer.query_count == before:
            stop_reason = "proposal_stall"
            break
    return _finalize(
        "adalead", archive, config, scorer,
        {"steps": len(trajectory), "parent_pool_multiplier": config.parent_pool_multiplier,
         "rollout": "accepted_child",
         "rollout_expected_mutations": config.adalead_mu,
         "rollout_max_depth": config.adalead_max_rollout,
         "rollout_allow_equal": config.adalead_allow_equal,
         "rollout_lineages": config.adalead_lineages or len(initial),
         "mutation_scheme": "poisson_space_native_operations",
         "stop_reason": stop_reason,
         "accepted_children": accepted_total,
         "accepted_rollout_steps": rollout_steps,
         "parent_selection": "rank_weighted_quality_with_novelty_exploration",
         "offspring_comparison": "rollout_root",
         "survivor_selection": "rank_weighted_quality_with_novelty_exploration",
         "exploration_fraction": config.adalead_exploration_fraction,
         "quality_rank_power": config.adalead_quality_power,
         "best_anchor_preserved": True}, trajectory
    )


def _configured_space(
    length, alphabet, config, algorithm, initial=None, *, cds_reference=None
):
    if not config.is_CDS_optimized:
        return _sequence_space(length, alphabet)
    if initial is None:
        raise ValueError("initial is required when is_CDS_optimized=True")
    from .codons import SynonymousCodonSpace

    values = normalize_nucleotide_sequences(initial)
    if not len(values):
        raise ValueError("initial must contain at least one CDS")
    reference = str(values[0]) if cds_reference is None else str(cds_reference)
    space = SynonymousCodonSpace(reference)
    requested = _sequence_space(length, alphabet)
    if requested.length != space.length or requested.alphabet != "ACGT":
        raise ValueError("length must match the initial CDS and alphabet must be ACGT")
    space.validate(values)
    return space


def _apply_fixed_region(space, fixed_region, initial, rng):
    positions = resolve_fixed_positions(fixed_region, space.length)
    if not positions:
        return space
    if initial is None:
        template = str(space.random(1, rng)[0])
    else:
        values = space.validate(initial)
        if not len(values):
            raise ValueError("initial must contain at least one sequence")
        template = str(values[0])
    restrict = getattr(space, "with_fixed_positions", None)
    if restrict is not None:
        space = restrict(template, positions)
    return FixedRegionSequenceSpace(space, template, positions)


def _run_budget_segment(
    name: str,
    scorer: SequenceScorer,
    length: int,
    config: OptimizationConfig,
    *,
    alphabet: str = "ACGT",
    initial: Sequence[str] | np.ndarray | None = None,
    objective: TargetObjective
    | MultiTaskObjective
    | list[str]
    | tuple[str, ...]
    | None = None,
    constraint: ConstraintSpec | None = None,
    seed: int = 0,
    count_repeated: bool = False,
    cds_reference: str | None = None,
) -> OptimizationResult:
    """Run one algorithm against a single- or multi-task activity scorer.

    The model output is scalarized by ``objective``. If omitted, output zero is
    maximized, which is the natural default for a single-task model. Direction
    lists use an equal-weight mean across active outputs.
    """
    key = str(name).strip().lower()
    space = _configured_space(
        length, alphabet, config, key, initial, cds_reference=cds_reference
    )
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown algorithm {name!r}; choose from {', '.join(available_algorithms())}"
        )
    rng = np.random.default_rng(seed)
    constraint_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0xC0DEC0DE])
    )
    random.seed(seed)
    space = _apply_fixed_region(space, config.fixed_region, initial, rng)
    prepared = _initial(initial, space, config, rng)
    objective_scorer: ObjectiveScorer | None
    if isinstance(scorer, CachedScorer):
        if objective is not None:
            raise ValueError(
                "pass an uncached activity scorer when using a multi-task objective"
            )
        budgeted = scorer
        if constraint is not None:
            budgeted.set_constraint(constraint, constraint_rng)
        if key == "random":
            budgeted.count_repeated = True
        objective_scorer = None
    else:
        objective_scorer = ObjectiveScorer(
            scorer,
            TargetObjective() if objective is None else objective,
        )
        proposal_count_algorithms = {
            "random",
            "ledidi",
            "vanilla_activation_maximization",
            "fast_seqprop",
            "den",
        }
        budgeted = CachedScorer(
            objective_scorer,
            config.budget,
            count_repeated=count_repeated or key in proposal_count_algorithms,
            constraint=constraint,
            constraint_rng=constraint_rng,
        )
    if budgeted.budget is None:
        budgeted.budget = config.budget
    prepared = _constraint_initial(prepared, budgeted, space, config, rng)
    prepared = _prediction_initial(prepared, budgeted, space, config, rng)
    result = _REGISTRY[key](prepared, budgeted, space, config, rng)
    fixed_positions = tuple(getattr(space, "fixed_positions", ()))
    if fixed_positions:
        result.metadata["fixed_positions"] = list(fixed_positions)
    if objective_scorer is not None:
        activities = objective_scorer.activities_for(result.sequences)
        result.attach_activities(
            activities,
            objective_scorer.resolved_task_names(activities.shape[1]),
        )
        result.metadata["objective"] = objective_scorer.objective.describe(
            activities.shape[1], objective_scorer.task_names
        )
        if isinstance(objective_scorer.objective, MultiTaskObjective):
            result.metadata["objective_evaluation"] = (
                objective_scorer.objective.evaluate(
                    activities, objective_scorer.task_names
                )
            )
    if budgeted.constraint_evaluator is not None:
        evaluator = budgeted.constraint_evaluator
        result.attach_constraint_scores(
            evaluator.probabilities_for(result.sequences), evaluator.names
        )
        result.metadata["constraint"] = {
            **evaluator.describe(),
            "screened_query_count": int(evaluator.evaluated_counts[0]),
            "counted_query_count": result.query_count,
            "model_query_count": budgeted.model_query_count,
        }
    return result


def optimize(
    model: SequenceScorer,
    length: int,
    initial: Sequence[str] | np.ndarray | None,
    objective: TargetObjective
    | MultiTaskObjective
    | list[str]
    | tuple[str, ...]
    | None,
    num_candidates: int,
    *,
    algorithm: str = "genetic_algorithm",
    stop_values: Sequence[float | tuple[float | None, float | None] | None] | None = None,
    stop_tolerance: float | None = 0.5,
    gain_fraction: float | None = None,
    gain_fractions: Sequence[float] | None = None,
    score_threshold: float | None = None,
    saturation: SaturationConfig | None = None,
    diversity: int = 5,
    fixed_region: Sequence[int | str | slice] | None = None,
    alphabet: str = "ACGT",
    constraint: ConstraintSpec | None = None,
    seed: int = 0,
    mutation_rate: float = 0.02,
    elite_fraction: float = 0.2,
    parent_pool_multiplier: int = 5,
    adalead_mu: float = 1.0,
    adalead_max_rollout: int = 8,
    adalead_allow_equal: bool = True,
    adalead_lineages: int | None = None,
    adalead_exploration_fraction: float = 0.2,
    adalead_quality_power: float = 4.0,
    ism_lineage_stopping: bool = True,
    ism_min_sweeps: int = 10,
    ism_patience: int = 5,
    ism_absolute_gain: float = 0.01,
    ism_relative_gain: float = 0.001,
    ism_knee_slope_ratio: float = 0.1,
    annealing_iterations: int = 1000,
    annealing_initial_temperature: float = 0.1,
    annealing_final_temperature: float = 0.05,
    learning_rate: float = 0.1,
    generator_learning_rate: float = 3e-3,
    generator_pretrain_epochs: int = 20,
    generator_update_epochs: int = 5,
    cross_entropy_batch_size: int = 256,
    den_seed_pretrain_epochs: int = 100,
    den_diversity_weight: float = 0.1,
    ledidi_edit_penalty: float = 0.1,
    ledidi_temperature: float = 1.0,
    ledidi_epsilon: float = 1e-4,
    basesearch_decay: float = 0.995,
    basesearch_queue_size: int = 1024,
    basesearch_max_stalled: int = 1024,
    basesearch_batch_size: int = 256,
    is_CDS_optimized: bool = False,
    count_repeated: bool = False,
) -> (
    OptimizationResult | SaturationResult | list[OptimizationResult | SaturationResult]
):
    """Optimize sequences with a parameterized model.

    ``diversity`` is the minimum pairwise Hamming distance among returned
    sequences. ``fixed_region`` holds zero-based positions or Python-style
    slices to one shared sequence template. ``stop_values`` supplies inclusive
    directional caps or (lower, upper) intervals; None means unlimited.
    Use ``score_threshold`` for stopping at a scalar score, or
    ``gain_fraction`` for selecting a fraction of the observed plateau gain.
    ``gain_fractions`` returns a SaturationResult with the requested archives
    from a single run, accessible by fraction.
    """
    if stop_values is not None:
        if isinstance(objective, (list, tuple)):
            objective = MultiTaskObjective(directions=objective, stop_values=stop_values, stop_tolerance=stop_tolerance)
        elif isinstance(objective, MultiTaskObjective):
            objective = replace(objective, stop_values=stop_values, stop_tolerance=stop_tolerance)
        else:
            raise ValueError("stop_values requires a directions objective")
    parameters = dict(
        num_candidates=num_candidates,
        diversity=diversity,
        fixed_region=fixed_region,
        mutation_rate=mutation_rate,
        elite_fraction=elite_fraction,
        parent_pool_multiplier=parent_pool_multiplier,
        adalead_mu=adalead_mu,
        adalead_max_rollout=adalead_max_rollout,
        adalead_allow_equal=adalead_allow_equal,
        adalead_lineages=adalead_lineages,
        adalead_exploration_fraction=adalead_exploration_fraction,
        adalead_quality_power=adalead_quality_power,
        ism_lineage_stopping=ism_lineage_stopping,
        ism_min_sweeps=ism_min_sweeps,
        ism_patience=ism_patience,
        ism_absolute_gain=ism_absolute_gain,
        ism_relative_gain=ism_relative_gain,
        ism_knee_slope_ratio=ism_knee_slope_ratio,
        annealing_iterations=annealing_iterations,
        annealing_initial_temperature=annealing_initial_temperature,
        annealing_final_temperature=annealing_final_temperature,
        learning_rate=learning_rate,
        generator_learning_rate=generator_learning_rate,
        generator_pretrain_epochs=generator_pretrain_epochs,
        generator_update_epochs=generator_update_epochs,
        cross_entropy_batch_size=cross_entropy_batch_size,
        den_seed_pretrain_epochs=den_seed_pretrain_epochs,
        den_diversity_weight=den_diversity_weight,
        ledidi_edit_penalty=ledidi_edit_penalty,
        ledidi_temperature=ledidi_temperature,
        ledidi_epsilon=ledidi_epsilon,
        basesearch_decay=basesearch_decay,
        basesearch_queue_size=basesearch_queue_size,
        basesearch_max_stalled=basesearch_max_stalled,
        basesearch_batch_size=basesearch_batch_size,
        is_CDS_optimized=is_CDS_optimized,
    )
    from .saturation import resolve_search_config

    algorithm_key = str(algorithm).strip().lower()
    explicit_saturation = saturation is not None or gain_fractions is not None
    saturation = resolve_search_config(
        saturation, gain_fraction, score_threshold, gain_fractions
    )
    if algorithm_key == "random":
        count_repeated = True

    def execute(one_initial, one_seed, *, cds_reference=None):
        from .saturation import _optimize_with_saturation

        saturated = _optimize_with_saturation(
            model, length, algorithm=algorithm, alphabet=alphabet,
            initial=one_initial, objective=objective, constraint=constraint,
            saturation=saturation, score_threshold=score_threshold, seed=one_seed,
            count_repeated=count_repeated, cds_reference=cds_reference, **parameters,
        )
        return saturated if explicit_saturation else saturated.selected

    if not is_CDS_optimized:
        return execute(initial, seed)
    if initial is None:
        raise ValueError("initial is required when is_CDS_optimized=True")
    cds_values = normalize_nucleotide_sequences(initial)
    if not len(cds_values):
        raise ValueError("initial must contain at least one CDS")
    from .codons import SynonymousCodonSpace

    prepared_cds = []
    seed_config = OptimizationConfig(
        budget=max(_DEFAULT_BUDGET, num_candidates),
        **parameters,
    )
    for offset, value in enumerate(cds_values):
        reference = str(value)
        space = SynonymousCodonSpace(reference)
        if space.length != length:
            raise ValueError("length must match every initial CDS")
        local_seed = seed + offset
        local_rng = np.random.default_rng(local_seed)
        space = _apply_fixed_region(space, fixed_region, [reference], local_rng)
        population = _initial(
            None,
            space,
            seed_config,
            local_rng,
            anchor=None if algorithm_key == "random" else reference,
        )
        prepared_cds.append((reference, population, local_seed))

    def optimize_cds(reference, population, local_seed):
        result = execute(population, local_seed, cds_reference=reference)
        selections = (
            [*result.selections.values(), result.knee, result.knee_fraction_result]
            if isinstance(result, SaturationResult)
            else (result,)
        )
        for selection in selections:
            selection.metadata["initial_cds"] = reference
        return result

    results = [optimize_cds(*values) for values in prepared_cds]
    return results[0] if len(results) == 1 else results


def _named_optimizer(name, public_name=None, doc=None):
    def function(model, length, initial, objective, num_candidates, **kwargs):
        if "algorithm" in kwargs:
            raise TypeError(f"{function.__name__}() does not accept algorithm")
        return optimize(
            model,
            length,
            initial,
            objective,
            num_candidates,
            algorithm=name,
            **kwargs,
        )

    function.__name__ = public_name or f"{name}_optimize"
    function.__qualname__ = function.__name__
    function.__module__ = __name__
    function.__doc__ = doc or f"Optimize sequences with {name}."
    parameters = [
        parameter
        for parameter in inspect.signature(optimize).parameters.values()
        if parameter.name != "algorithm"
    ]
    function.__signature__ = inspect.signature(optimize).replace(parameters=parameters)
    return function


basesearch_optimize = _named_optimizer(
    "basesearch",
    "basesearch_optimize",
    "Priority-decayed black-box search with unique mutation proposals.",
)
random_optimize = _named_optimizer(
    "random",
    "random_optimize",
    "Uniform random sampling with plateau or threshold stopping.",
)
ism_optimize = _named_optimizer("ism_greedy", "ism_optimize")
simulated_annealing_optimize = _named_optimizer(
    "simulated_annealing", "simulated_annealing_optimize"
)
genetic_algorithm_optimize = _named_optimizer(
    "genetic_algorithm", "genetic_algorithm_optimize"
)
adalead_optimize = _named_optimizer("adalead", "adalead_optimize")
beam_search_optimize = _named_optimizer("beam_search", "beam_search_optimize")
cross_entropy_optimize = _named_optimizer(
    "cross_entropy_method", "cross_entropy_optimize"
)
vanilla_activation_maximization_optimize = _named_optimizer(
    "vanilla_activation_maximization", "vanilla_activation_maximization_optimize"
)
fast_seqprop_optimize = _named_optimizer("fast_seqprop", "fast_seqprop_optimize")
ledidi_optimize = _named_optimizer(
    "ledidi",
    "ledidi_optimize",
    "Ledidi-style single-output maximization with an edit penalty.",
)
den_optimize = _named_optimizer("den", "den_optimize")


# Requested spelling retained as a public alias; lower-case ``ism_optimize`` is
# preferred in Python code.
ISM_optimize = ism_optimize


# Import optional advanced implementations after the registry and common
# helpers are defined. Importing this module registers their public names.
from . import advanced as _advanced  # noqa: E402,F401
