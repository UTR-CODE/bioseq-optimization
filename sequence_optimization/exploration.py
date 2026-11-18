"""Quality-gated cooperative search over caller-defined motif/position programs.

Local optimizers retain their original scalar objective. Between bounded search
segments they exchange seeds through a shared archive; targeted motif proposals
explore underrepresented positions. Only discrete, feasible predictions enter
the archive. This module never treats a soft PWM as a validated candidate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .api import SequenceSpace, normalize_nucleotide_sequences
from .constraints import ConstraintEvaluator
from .objectives import MultiTaskObjective, TargetObjective


class MotifPositionDescriptor:
    """Exact observed variants grouped by motif identity, position and copy count.

    By default reverse complements belong to the same motif. Set
    ``reverse_complements=False`` for directional RNA patterns. ``include_strand=True`` keeps
    their orientation as a separate feature, not as a separate motif identity.
    Bins use hit midpoints in the supplied sequence orientation. This is a search
    descriptor, not a motif discovery algorithm or a TF-binding annotation.
    """

    def __init__(
        self,
        motifs: Mapping[str, Sequence[str]],
        *,
        n_bins=10,
        include_strand=False,
        max_count=2,
        reverse_complements=True,
    ):
        for name, value in (("n_bins", n_bins), ("max_count", max_count)):
            _positive_integer(name, value)
        if not isinstance(include_strand, bool):
            raise ValueError("include_strand must be boolean")
        if not isinstance(reverse_complements, bool):
            raise ValueError("reverse_complements must be boolean")
        if include_strand and not reverse_complements:
            raise ValueError("include_strand requires reverse_complements=True")
        self.reverse_complements = reverse_complements
        if not isinstance(motifs, Mapping) or not motifs:
            raise ValueError("motifs must be a nonempty name-to-variants mapping")
        self.n_bins, self.include_strand, self.max_count = (
            n_bins,
            include_strand,
            max_count,
        )
        self.motifs: dict[str, tuple[str, ...]] = {}
        self._lookup: dict[str, set[tuple[int, int]]] = defaultdict(set)
        self._complement = str.maketrans("ACGT", "TGCA")
        owners = {}
        for index, (name, variants) in enumerate(motifs.items()):
            if (
                not isinstance(name, str)
                or not name.strip()
                or isinstance(variants, str)
            ):
                raise ValueError(
                    "each motif needs a nonempty name and a sequence of variants"
                )
            words = tuple(
                sorted(set(str(v).upper().replace("U", "T") for v in variants))
            )
            if not words or any(not w or set(w) - set("ACGT") for w in words):
                raise ValueError("motif variants must be nonempty A/C/G/T strings")
            self.motifs[name] = words
            for word in words:
                rc = self.reverse_complement(word)
                canonical = min(word, rc) if reverse_complements else word
                if canonical in owners and owners[canonical] != name:
                    raise ValueError(
                        "identical/reverse-complement variants cannot belong to different motifs"
                    )
                owners[canonical] = name
                # Canonical orientation is deterministic, independent of variant order.
                self._lookup[canonical].add((index, 0))
                if reverse_complements:
                    reverse = self.reverse_complement(canonical)
                    self._lookup[reverse].add(
                        (index, int(include_strand and canonical != reverse))
                    )
        self._lengths = sorted(set(map(len, self._lookup)))
        self.width = len(self.motifs) * n_bins * (2 if include_strand else 1)
        self.feature_names = tuple(
            f"{name}:bin{b + 1}:strand{s}"
            for name in self.motifs
            for b in range(n_bins)
            for s in range(2 if include_strand else 1)
        )

    def reverse_complement(self, word):
        return word.translate(self._complement)[::-1]

    def __call__(self, sequences):
        values = normalize_nucleotide_sequences(sequences)
        if not len(values):
            return np.zeros((0, self.width), dtype=np.uint8)
        SequenceSpace(len(values[0])).validate(values)
        if max(self._lengths) > len(values[0]):
            raise ValueError("motif variants cannot exceed sequence length")
        features = np.zeros((len(values), self.width), dtype=np.int64)
        strands = 2 if self.include_strand else 1
        for row, seq in enumerate(values):
            for k in self._lengths:
                for start in range(len(seq) - k + 1):
                    for motif, strand in self._lookup.get(seq[start : start + k], ()):
                        b = min(
                            self.n_bins - 1, (start + k // 2) * self.n_bins // len(seq)
                        )
                        features[row, (motif * self.n_bins + b) * strands + strand] += 1
        return np.minimum(features, self.max_count)

    def propose(self, parents, coverage, rng, fixed_positions=()):
        """Replace equal-length segments at sparsely covered motif/bin features."""
        parents = list(parents)
        weights = 1.0 / (1.0 + np.asarray(coverage, dtype=float))
        fixed = set(fixed_positions)
        strands = 2 if self.include_strand else 1
        variants = list(self.motifs.values())
        result = []
        for parent in parents:
            # Bounded attempts handle short sequences and fixed regions.
            for _ in range(32):
                feature = int(rng.choice(self.width, p=weights / weights.sum()))
                motif, remainder = divmod(feature, self.n_bins * strands)
                b, strand = divmod(remainder, strands)
                word = str(rng.choice(variants[motif]))
                if self.reverse_complements:
                    canonical = min(word, self.reverse_complement(word))
                    word = self.reverse_complement(canonical) if strand else canonical
                    if not self.include_strand and rng.random() < 0.5:
                        word = self.reverse_complement(word)
                starts = [
                    i
                    for i in range(len(parent) - len(word) + 1)
                    if (i + len(word) // 2) * self.n_bins // len(parent) == b
                    and not fixed.intersection(range(i, i + len(word)))
                ]
                if starts:
                    start = int(rng.choice(starts))
                    child = parent[:start] + word + parent[start + len(word) :]
                    if child != parent:
                        result.append(child)
                        break
            else:
                result.append(parent)
        return np.asarray(result, dtype=str)


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DiversitySearchConfig:
    """One shared discrete-proposal budget, including seeds and cache hits.

    The quality floor is max(min_quality, baseline + quality_fraction * observed
    gain), where baseline is the feasible initial mean and gain is the best
    motif-bearing quality minus baseline, clipped at zero. It never decreases.
    ``quality_fn`` optionally converts raw [N,T] activities to a common higher-is-
    better quality; otherwise the unmodified scalar objective is used.
    """

    total_budget: int = 100_000
    segment_budget: int = 2_000
    quality_fraction: float = 0.8
    min_quality: float | None = None
    quality_fn: Callable | None = field(default=None, repr=False)
    activity_bounds: Mapping[int | str, tuple[float | None, float | None]] = field(
        default_factory=dict
    )
    max_patterns: int = 512
    per_pattern: int = 4
    motif_proposal_fraction: float = 0.2
    original_seed_fraction: float = 0.25
    exchange_seeds: bool = True

    def __post_init__(self):
        for name in ("total_budget", "segment_budget", "max_patterns", "per_pattern"):
            _positive_integer(name, getattr(self, name))
        for name in (
            "quality_fraction",
            "motif_proposal_fraction",
            "original_seed_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_quality is not None and not np.isfinite(self.min_quality):
            raise ValueError("min_quality must be finite or None")
        if self.quality_fn is not None and not callable(self.quality_fn):
            raise TypeError("quality_fn must be callable")
        if not isinstance(self.exchange_seeds, bool):
            raise ValueError("exchange_seeds must be boolean")
        for bounds in self.activity_bounds.values():
            if len(bounds) != 2 or any(
                v is not None and not np.isfinite(v) for v in bounds
            ):
                raise ValueError(
                    "activity_bounds entries must be finite (lower, upper) pairs"
                )
            if all(v is not None for v in bounds) and bounds[0] > bounds[1]:
                raise ValueError("activity_bounds lower must not exceed upper")


@dataclass
class DiversitySearchResult:
    candidates: pd.DataFrame
    archive: pd.DataFrame
    history: pd.DataFrame
    metadata: dict[str, Any]

    def shortlist(self, *, source="candidates", **kwargs):
        """Cluster the candidate panel or larger archive and recommend a small panel.

        By default retain this search's quality floor and minimum Hamming distance.
        Extra feature blocks must align to the selected source DataFrame.
        """
        from .panel import cluster_designs

        if source not in {"candidates", "archive"}:
            raise ValueError("source must be candidates or archive")
        kwargs.setdefault("min_quality", self.metadata["quality_floor"])
        kwargs.setdefault("min_hamming", self.metadata["diversity"])
        descriptor = kwargs.pop("descriptor", None)
        if descriptor is None:
            descriptor = MotifPositionDescriptor(**self.metadata["descriptor"])
        return cluster_designs(getattr(self, source), descriptor=descriptor, **kwargs)

    def save(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name in ("candidates", "archive", "history"):
            getattr(self, name).to_csv(root / f"{name}.csv", index=False)
        (root / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
        with (root / "candidates.fa").open("w") as out:
            for i, seq in enumerate(self.candidates.sequence, 1):
                out.write(f">design_{i}\n{seq}\n")


class _PatternArchive:
    def __init__(self, descriptor, config, objective, task_names):
        self.descriptor, self.config, self.objective = descriptor, config, objective
        self.task_names = task_names
        self.entries: dict[str, dict] = {}
        self.baseline = None
        self.peak = -np.inf
        self.visits = Counter()
        self._bounds = None

    def feasible(self, activities):
        passed = (
            self.objective.feasible(activities)
            if isinstance(self.objective, MultiTaskObjective)
            else np.ones(len(activities), dtype=bool)
        )
        if self._bounds is None:
            self._bounds = {}
            for name, bounds in self.config.activity_bounds.items():
                if isinstance(name, str):
                    if self.task_names is None or name not in self.task_names:
                        raise ValueError(f"unknown activity bound task {name!r}")
                    index = self.task_names.index(name)
                else:
                    if isinstance(name, bool) or not isinstance(
                        name, (int, np.integer)
                    ):
                        raise ValueError(
                            "activity bound keys must be task names or indices"
                        )
                    index = int(name)
                if not 0 <= index < activities.shape[1]:
                    raise ValueError("activity bound index outside model outputs")
                self._bounds[index] = bounds
        for i, (low, high) in self._bounds.items():
            if low is not None:
                passed &= activities[:, i] >= low
            if high is not None:
                passed &= activities[:, i] <= high
        return passed

    @property
    def floor(self):
        baseline = 0.0 if self.baseline is None else self.baseline
        floor = baseline + self.config.quality_fraction * max(0.0, self.peak - baseline)
        return max(
            floor,
            -np.inf if self.config.min_quality is None else self.config.min_quality,
        )

    def observe(self, sequences, batch, source, *, initialize=False):
        activities = batch.scores
        if activities.ndim == 1:
            activities = activities[:, None]
        scores = self.objective.scalarize(activities, self.task_names)
        qualities = (
            scores
            if self.config.quality_fn is None
            else np.asarray(self.config.quality_fn(activities.copy()), dtype=float)
        )
        if qualities.shape != (len(sequences),) or not np.isfinite(qualities).all():
            raise ValueError("quality_fn must return a finite [N] vector")
        passed = self.feasible(activities)
        if initialize:
            if (
                isinstance(self.objective, MultiTaskObjective)
                and not self.objective.feasible(activities).all()
            ):
                raise ValueError("initial sequences must satisfy objective bounds")
            self.baseline = float(qualities.mean())
        features = self.descriptor(sequences)
        structured = features.any(axis=1) & passed
        if structured.any():
            self.peak = max(self.peak, float(qualities[structured].max()))
        for i in np.flatnonzero(structured & (qualities >= self.floor)):
            seq = str(sequences[i])
            if seq in self.entries:
                self.entries[seq]["sources"].add(source)
                continue
            signature = tuple(int(v) for v in features[i])
            self.entries[seq] = dict(
                sequence=seq,
                score=float(scores[i]),
                quality=float(qualities[i]),
                features=signature,
                activities=activities[i].copy(),
                sources={source},
                first_source=source,
            )
        self.prune()

    def prune(self):
        groups = defaultdict(list)
        for entry in self.entries.values():
            if entry["quality"] >= self.floor:
                groups[entry["features"]].append(entry)
        ranked = sorted(
            groups, key=lambda key: (-max(e["quality"] for e in groups[key]), key)
        )
        self.entries = {
            e["sequence"]: e
            for key in ranked[: self.config.max_patterns]
            for e in sorted(groups[key], key=lambda e: (-e["quality"], e["sequence"]))[
                : self.config.per_pattern
            ]
        }

    def coverage(self):
        if not self.entries:
            return np.zeros(self.descriptor.width, dtype=int)
        return (np.asarray([e["features"] for e in self.entries.values()]) > 0).sum(
            axis=0
        )

    def select(self, count, *, min_hamming=0, feedback=False):
        """Quality-gated max-min weighted-Jaccard selection, with global spacing."""
        entries = sorted(
            self.entries.values(), key=lambda e: (-e["quality"], e["sequence"])
        )
        if not entries:
            return []
        features = np.asarray([e["features"] for e in entries], dtype=float)
        codes = np.asarray(
            [list(e["sequence"].encode()) for e in entries], dtype=np.uint8
        )
        novelty = np.ones(len(entries))
        hamming = np.full(len(entries), codes.shape[1] + 1)
        available = np.ones(len(entries), dtype=bool)
        selected = []
        visits = self.visits.copy()
        for _ in range(min(count, len(entries))):
            ids = np.flatnonzero(available & (hamming >= min_hamming))
            if not len(ids):
                break
            if feedback:
                chosen = min(
                    ids,
                    key=lambda i: (
                        visits[entries[i]["features"]],
                        -novelty[i],
                        -entries[i]["quality"],
                        -hamming[i],
                        entries[i]["sequence"],
                    ),
                )
            else:
                chosen = min(
                    ids,
                    key=lambda i: (
                        -novelty[i],
                        -entries[i]["quality"],
                        -hamming[i],
                        entries[i]["sequence"],
                    ),
                )
            entry = entries[chosen]
            selected.append(entry)
            available[chosen] = False
            if feedback:
                visits[entry["features"]] += 1
            union = np.maximum(features, features[chosen]).sum(axis=1)
            distances = 1 - np.minimum(features, features[chosen]).sum(axis=1) / union
            novelty = np.minimum(novelty, distances)
            hamming = np.minimum(hamming, (codes != codes[chosen]).sum(axis=1))
        return selected

    def frame(self, entries=None):
        entries = list(self.entries.values()) if entries is None else entries
        names = self.task_names
        if names is None and entries:
            names = tuple(f"task_{i}" for i in range(len(entries[0]["activities"])))
        patterns = sorted({e["features"] for e in self.entries.values()})
        ids = {p: f"P{i + 1:04d}" for i, p in enumerate(patterns)}
        rows = []
        for entry in sorted(entries, key=lambda e: (-e["quality"], e["sequence"])):
            rows.append(
                dict(
                    sequence=entry["sequence"],
                    algorithm=";".join(sorted(entry["sources"])),
                    first_source=entry["first_source"],
                    score=entry["score"],
                    quality=entry["quality"],
                    pattern=ids[entry["features"]],
                    motif_pattern=";".join(
                        f"{name}={v}"
                        for name, v in zip(
                            self.descriptor.feature_names, entry["features"]
                        )
                        if v
                    ),
                    **{
                        f"activity_{n}": float(v)
                        for n, v in zip(names, entry["activities"])
                    },
                )
            )
        return pd.DataFrame(
            rows,
            columns=[
                "sequence",
                "algorithm",
                "score",
                "quality",
                "pattern",
                "first_source",
                "motif_pattern",
            ]
            + [f"activity_{n}" for n in names or ()],
        )


class _ArchiveScorer:
    def __init__(self, shared, archive, source, soft_model=None):
        self.shared, self.archive, self.source = shared, archive, source
        self.soft_model = shared if soft_model is None else soft_model
        self.task_names = archive.task_names
        self.soft_queries = 0

    def __getattr__(self, name):
        return getattr(self.soft_model, name)

    def score(self, sequences):
        batch = self.shared.score(sequences)
        self.archive.observe(sequences, batch, self.source)
        return batch

    def score_soft(self, probabilities):
        self.soft_queries += len(probabilities)
        return self.soft_model.score_soft(probabilities)


def generate_diverse_designs(
    model,
    length,
    initial,
    objective,
    *,
    descriptor: MotifPositionDescriptor,
    config: DiversitySearchConfig | None = None,
    algorithms=("genetic_algorithm", "adalead", "cross_entropy_method", "ism_greedy"),
    num_candidates=100,
    diversity=5,
    seed=0,
    weights=None,
    stop_values=None,
    stop_tolerance=0.5,
    tasks=None,
    constraint=None,
    fixed_region=None,
    differentiable_model=None,
    algorithm_options: Mapping[str, Mapping] | None = None,
) -> DiversitySearchResult:
    """Search cooperatively under a total budget and return a global assay panel.

    Each segment restarts an existing optimizer with quality-gated, motif-diverse
    shared seeds and some original seeds. Optimizer state is not preserved between
    segments. All scored feasible hard candidates, not only local top-N outputs,
    can enter the archive. Targeted insertions replace equal-length mutable segments.
    Seeds must satisfy sequence constraints and objective bounds; archive-only
    activity_bounds may be unmet initially. No CDS synonymy mode is
    supported here; use the existing CDS optimizers for that search space.
    """
    from .algorithms import _run_budget_segment, available_algorithms
    from .api import OptimizationConfig
    from .api import FixedRegionSequenceSpace, resolve_fixed_positions
    from .suite import (
        DIFFERENTIABLE_ALGORITHMS,
        _SharedScorer,
        _TaskNamedScorer,
        _supports_soft_scoring,
    )

    config = DiversitySearchConfig() if config is None else config
    if not isinstance(config, DiversitySearchConfig) or not isinstance(
        descriptor, MotifPositionDescriptor
    ):
        raise TypeError(
            "config/descriptor must be DiversitySearchConfig/MotifPositionDescriptor"
        )
    _positive_integer("num_candidates", num_candidates)
    if (
        isinstance(diversity, bool)
        or not isinstance(diversity, (int, np.integer))
        or not 0 <= diversity <= length
    ):
        raise ValueError("diversity must be an integer between zero and length")
    if isinstance(algorithms, str):
        raise TypeError("algorithms must be a sequence of names")
    names = tuple(str(n).strip().lower() for n in algorithms)
    if (
        not names
        or len(set(names)) != len(names)
        or set(names) - set(available_algorithms())
    ):
        raise ValueError("algorithms must contain unique registered names")
    options = {} if algorithm_options is None else dict(algorithm_options)
    if set(options) - set(names):
        raise ValueError("algorithm_options contains an unselected algorithm")
    forbidden = {
        "algorithm",
        "budget",
        "saturation",
        "count_repeated",
        "diversity",
        "seed",
        "constraint",
        "fixed_region",
        "is_CDS_optimized",
        "alphabet",
        "stop_values",
        "stop_tolerance",
        "length",
        "initial",
        "objective",
        "num_candidates",
        "model",
    }
    if any(forbidden.intersection(opts) for opts in options.values()):
        raise ValueError(
            "algorithm_options cannot override shared search/budget/constraint settings"
        )
    space = SequenceSpace(length)
    initial = space.validate(initial)
    if not len(initial) or len(set(initial)) != len(initial):
        raise ValueError("initial must contain unique, nonempty seed sequences")
    # Validate the descriptor and fixed positions before any model call.
    descriptor(initial[:1])
    if fixed_region is not None:
        positions = resolve_fixed_positions(fixed_region, length)
        space = FixedRegionSequenceSpace(space, str(initial[0]), positions)
        space.validate(initial)
    population = len(initial)
    if config.segment_budget < 3 * population:
        raise ValueError(
            "segment_budget must be at least three times the seed population"
        )
    if config.total_budget < population + 3 * population * len(names):
        raise ValueError(
            "total_budget must allow initialization and one three-population segment per algorithm"
        )
    if objective is None:
        objective = TargetObjective()
    if isinstance(objective, (list, tuple)):
        objective = MultiTaskObjective(
            directions=objective,
            weights=weights,
            stop_values=stop_values,
            stop_tolerance=stop_tolerance,
        )
    elif isinstance(objective, MultiTaskObjective):
        updates = {}
        if weights is not None:
            updates["weights"] = weights
        if stop_values is not None:
            updates.update(stop_values=stop_values, stop_tolerance=stop_tolerance)
        objective = replace(objective, **updates)
    elif (
        not isinstance(objective, TargetObjective)
        or weights is not None
        or stop_values is not None
    ):
        raise ValueError("weights/stop_values require a multi-task objective")
    if tasks is not None:
        task_names = tuple(str(n).strip() for n in tasks)
        if (
            not task_names
            or any(not n for n in task_names)
            or len(set(task_names)) != len(task_names)
        ):
            raise ValueError("tasks must be unique nonempty names")
        model = _TaskNamedScorer(model, task_names)
    gradient = model if differentiable_model is None else differentiable_model
    if set(names) & DIFFERENTIABLE_ALGORITHMS and not _supports_soft_scoring(gradient):
        raise TypeError("gradient algorithms require a model implementing score_soft")
    task_names = getattr(model, "task_names", None)
    task_names = None if task_names is None else tuple(task_names)
    archive = _PatternArchive(descriptor, config, objective, task_names)
    shared = _SharedScorer(model)
    rng = np.random.default_rng(seed)
    evaluator = (
        None
        if constraint is None
        else ConstraintEvaluator(constraint, np.random.default_rng(seed))
    )

    def hard_constraint(values):
        return evaluator.evaluate(values).astype(float)

    if evaluator is not None and not evaluator.evaluate(initial).all():
        raise ValueError("initial sequences must satisfy sequence constraints")
    initial_batch = shared.score(initial)
    archive.observe(initial, initial_batch, "initial", initialize=True)
    spent = population
    history = []
    soft_queries = 0
    previous = {}
    schedule = []
    segment = 0
    while config.total_budget - spent >= 3 * population:
        if not schedule:
            schedule = list(rng.permutation(names))
        name = str(schedule.pop())
        # Reserve a minimum segment for each still-unvisited algorithm.
        unvisited = set(names) - set(previous) - {name}
        available = config.total_budget - spent - 3 * population * len(unvisited)
        allocation = min(config.segment_budget, available)
        if allocation < 3 * population:
            break
        before_patterns = {e["features"] for e in archive.entries.values()}
        before_queries = len(shared._cache)
        retain = int(np.ceil(population * config.original_seed_fraction))
        seeds = list(rng.permutation(initial)[:retain])
        if config.exchange_seeds:
            seeds.extend(
                e["sequence"] for e in archive.select(population, feedback=True)
            )
        seeds.extend(previous.get(name, ()))
        seeds.extend(rng.permutation(initial))
        seeds = np.asarray(list(dict.fromkeys(map(str, seeds)))[:population], dtype=str)
        inject_count = min(
            int(allocation * config.motif_proposal_fraction),
            allocation - 3 * population,
        )
        if inject_count:
            parents = seeds[rng.integers(population, size=inject_count)]
            proposals = descriptor.propose(
                parents, archive.coverage(), rng, getattr(space, "fixed_positions", ())
            )
            passed = (
                np.ones(len(proposals), dtype=bool)
                if evaluator is None
                else evaluator.evaluate(proposals)
            )
            if passed.any():
                archive.observe(
                    proposals[passed],
                    shared.score(proposals[passed]),
                    f"{name}:motif_proposal",
                )
            spent += inject_count
            # Make targeted proposals available to the local optimizer immediately.
            if config.exchange_seeds:
                seeds = np.asarray(
                    list(
                        dict.fromkeys(
                            [
                                *seeds[:retain],
                                *(
                                    e["sequence"]
                                    for e in archive.select(population, feedback=True)
                                ),
                                *seeds,
                            ]
                        )
                    )[:population],
                    dtype=str,
                )
        wrapped = _ArchiveScorer(shared, archive, name, gradient)
        original_set, previous_set = set(initial), set(previous.get(name, ()))
        exchanged = sum(
            s not in original_set
            and s not in previous_set
            and s in archive.entries
            and archive.entries[s]["first_source"].split(":", 1)[0]
            not in {name, "initial"}
            for s in seeds
        )
        for seq in seeds:
            if seq in archive.entries:
                archive.visits[archive.entries[seq]["features"]] += 1
        result = _run_budget_segment(
            name,
            wrapped,
            length,
            OptimizationConfig(
                budget=allocation - inject_count, num_candidates=population,
                diversity=0, fixed_region=fixed_region, **options.get(name, {}),
            ),
            initial=seeds,
            objective=objective,
            seed=int(rng.integers(0, 2**31 - 1)),
            count_repeated=True,
            constraint=None if evaluator is None else hard_constraint,
        )
        spent += result.query_count
        soft_queries += wrapped.soft_queries
        previous[name] = result.sequences.tolist()
        after_patterns = {e["features"] for e in archive.entries.values()}
        history.append(
            dict(
                segment=segment,
                algorithm=name,
                query_count=spent,
                segment_queries=result.query_count + inject_count,
                motif_proposals=inject_count,
                model_queries=len(shared._cache) - before_queries,
                soft_queries=wrapped.soft_queries,
                exchanged_seeds=exchanged,
                archive_size=len(archive.entries),
                patterns=len(after_patterns),
                new_patterns=len(after_patterns - before_patterns),
                quality_floor=archive.floor,
                best_quality=None if not np.isfinite(archive.peak) else archive.peak,
            )
        )
        segment += 1
    selected = archive.select(num_candidates, min_hamming=diversity)
    candidates = archive.frame(selected)
    all_entries = archive.frame()
    coverage = archive.coverage()
    metadata = dict(
        mode="cooperative_motif_position",
        seed=int(seed),
        algorithms=list(names),
        task_names=list(task_names) if task_names is not None else None,
        objective=objective.describe(
            initial_batch.scores.shape[1] if initial_batch.scores.ndim == 2 else 1,
            task_names,
        ),
        total_budget=int(config.total_budget),
        query_count=int(spent),
        unused_budget=int(config.total_budget - spent),
        unique_model_queries=len(shared._cache),
        soft_sequence_evaluations=int(soft_queries),
        requested_candidates=int(num_candidates),
        returned_candidates=len(candidates),
        shortfall=int(num_candidates - len(candidates)),
        quality_baseline=archive.baseline,
        quality_floor=archive.floor,
        best_quality=None if not np.isfinite(archive.peak) else archive.peak,
        quality_fraction=float(config.quality_fraction),
        min_quality=config.min_quality,
        quality_definition="scalar_objective"
        if config.quality_fn is None
        else "custom_quality_fn",
        exchange_seeds=config.exchange_seeds,
        motif_proposal_fraction=float(config.motif_proposal_fraction),
        per_algorithm_queries={
            n: int(sum(r["segment_queries"] for r in history if r["algorithm"] == n))
            for n in names
        },
        activity_bounds={str(k): list(v) for k, v in config.activity_bounds.items()},
        diversity=int(diversity),
        archive_patterns=int(all_entries.pattern.nunique()),
        selected_patterns=int(candidates.pattern.nunique()),
        covered_features=[n for n, c in zip(descriptor.feature_names, coverage) if c],
        descriptor=dict(
            motifs={k: list(v) for k, v in descriptor.motifs.items()},
            n_bins=int(descriptor.n_bins),
            include_strand=descriptor.include_strand,
            reverse_complements=descriptor.reverse_complements,
            max_count=int(descriptor.max_count),
        ),
        archive_limit=int(config.max_patterns * config.per_pattern),
        provenance="algorithm lists hard-score observers, including inherited seeds; first_source is first retained admission",
        budget_semantics="discrete proposals including seed scoring and repeats; soft passes reported separately",
        stop_reason="budget_or_segment_remainder",
    )
    return DiversitySearchResult(
        candidates, all_entries, pd.DataFrame(history), metadata
    )
