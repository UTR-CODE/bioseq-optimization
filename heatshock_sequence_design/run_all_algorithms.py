"""Run every sequence-optimization algorithm for a DeepAdapt heat-shock design.

This script is intentionally self-contained in ``heatshock_sequence_design``.
It loads the local DeepAdapt checkpoint, optimizes the supplied 5'UTR, and
writes one CSV plus metadata JSON per algorithm.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Keep all project-specific imports local to this work directory.  The source
# checkouts are read-only dependencies; outputs are written beside this file.
WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parent
BIOSEQ_MODELS_ROOT = Path("/home/hexin/bioseq-models")
WEIGHTS_DIR = Path("/mnt/data/home/hexin/bioseq-models-weight-cache/0.10.0")
for path in (REPO_ROOT, BIOSEQ_MODELS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from bioseq_models import load_model
from bioseq_models.adapters.common import encode_acgt_sequences
from sequence_optimization import SaturationConfig, available_algorithms
from sequence_optimization.saturation import _optimize_with_saturation


REFERENCE_UTR5 = "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACC"
FIXED_KOZAK_TAIL = REFERENCE_UTR5[-6:]
OPTIMIZED_PREFIX_LENGTH = len(REFERENCE_UTR5) - len(FIXED_KOZAK_TAIL)
OBJECTIVE = ["maximize", "minimize", "maximize"]
NUM_CANDIDATES = 4
DIVERSITY = 3
SATURATION_FRACTION = 0.9
SATURATION = SaturationConfig(
    fractions=(SATURATION_FRACTION,),
    minimum_queries=40,
    plateau_patience_queries=80,
    maximum_queries=None,
    material_improvement=1e-4,
)


def read_fasta(path: Path) -> str:
    """Read the first FASTA record and normalize U to T."""
    sequence = "".join(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(">")
    ).upper().replace("U", "T")
    if not sequence or set(sequence) - set("ACGT"):
        raise ValueError(f"{path} does not contain a non-empty ACGT sequence")
    return sequence


def _probability(ok: bool) -> float:
    return 1.0 if ok else 0.0


def no_atg(sequence: str) -> float:
    return _probability("ATG" not in sequence.upper())


def fixed_kozak_tail(sequence: str) -> float:
    # The supplied UTR ends in GACACC; keep these six nucleotides unchanged.
    return _probability(sequence.upper().endswith(FIXED_KOZAK_TAIL))


def gc_fraction(sequence: str) -> float:
    value = sequence.upper()
    fraction = sum(base in "GC" for base in value) / max(len(value), 1)
    return _probability(0.20 <= fraction <= 0.80)


def no_homopolymer_of_six(sequence: str) -> float:
    return _probability(re.search(r"([ACGT])\1{5,}", sequence.upper()) is None)


def no_extreme_repeats(sequence: str) -> float:
    """Reject excessive repeated 4--8-mers anywhere in the UTR."""
    value = sequence.upper()
    for size in range(4, 9):
        counts = Counter(value[index : index + size] for index in range(len(value) - size + 1))
        if counts and max(counts.values()) >= 4:
            return 0.0
    return 1.0


def no_simple_repeats(sequence: str) -> float:
    """Reject tandem mono/di/tri-nucleotide repeats of four or more copies."""
    return _probability(
        re.search(r"([ACGT]{1,3})\1{3,}", sequence.upper()) is None
    )


CONSTRAINTS = [
    no_atg,
    fixed_kozak_tail,
    gc_fraction,
    no_homopolymer_of_six,
    no_extreme_repeats,
    no_simple_repeats,
]


def constraints_pass(sequence: str) -> bool:
    return all(function(sequence) >= 1.0 for function in CONSTRAINTS)


def _constraint_on_full_utr(function):
    """Evaluate a full-UTR constraint on an optimized 44-nt prefix."""

    def constraint(prefixes):
        if isinstance(prefixes, str):
            return function(prefixes + FIXED_KOZAK_TAIL)
        values = np.asarray(prefixes, dtype=str).reshape(-1)
        return np.asarray(
            [function(str(prefix) + FIXED_KOZAK_TAIL) for prefix in values],
            dtype=np.float64,
        )

    constraint.__name__ = function.__name__
    return constraint


OPTIMIZER_CONSTRAINTS = [_constraint_on_full_utr(function) for function in CONSTRAINTS]


def make_initial(count: int = NUM_CANDIDATES) -> list[str]:
    """Create deterministic valid seeds while preserving the fixed tail."""
    rng = np.random.default_rng(20260915)
    result: list[str] = []
    while len(result) < count:
        candidate = list(REFERENCE_UTR5)
        # Mutate only the first 44 bases.  The modest mutation rate keeps the
        # seeds near the supplied sequence while yielding distinct candidates.
        for position in rng.choice(OPTIMIZED_PREFIX_LENGTH, size=4, replace=False):
            choices = np.asarray(list("ACGT"))
            choices = choices[choices != candidate[position]]
            candidate[position] = str(rng.choice(choices))
        sequence = "".join(candidate)
        prefix = sequence[:OPTIMIZED_PREFIX_LENGTH]
        if prefix not in result and constraints_pass(sequence):
            result.append(prefix)
    return result


class DeepAdaptDifferentiableScorer:
    """DeepAdapt adapter with a differentiable path for gradient algorithms."""

    task_names = ("delta_log2_te", "normal_log2_te", "heatshock_log2_te")

    def __init__(self, model, *, utr5_length: int):
        self.model = model
        self.device = model.device
        self.utr5_length = utr5_length

    def score(self, sequences):
        values = np.asarray(sequences, dtype=str).reshape(-1)
        return self.model.score(
            np.asarray([str(value) + FIXED_KOZAK_TAIL for value in values], dtype=str)
        )

    def _soft_towers(self, probabilities: torch.Tensor):
        if probabilities.ndim != 3 or probabilities.shape[1] != self.utr5_length:
            raise ValueError(
                f"expected soft 5'UTR input [N, {self.utr5_length}, 4], "
                f"got {tuple(probabilities.shape)}"
            )
        batch = probabilities.shape[0]
        values = probabilities.transpose(1, 2)
        dtype, device = values.dtype, values.device
        utr5 = torch.zeros((batch, 4, 1030), dtype=dtype, device=device)
        utr5[:, :, 1000 - self.utr5_length : 1000] = values
        utr5_fixed = encode_acgt_sequences(
            [self.model.cds[:30]], exact_length=30
        ).to(device=device, dtype=dtype)
        utr5[:, :, 1000:1030] = utr5_fixed.expand(batch, -1, -1)

        utr3 = encode_acgt_sequences(
            [self.model.cds[-30:] + self.model.utr3], padded_length=4030
        ).to(device=device, dtype=dtype)
        utr3 = utr3.expand(batch, -1, -1).clone()

        cds_prefix = values[:, :, -30:]
        cds = encode_acgt_sequences(
            [self.model.cds], exact_length=len(self.model.cds)
        ).to(device=device, dtype=dtype)
        cds_suffix = encode_acgt_sequences(
            [self.model.utr3[:30]], exact_length=30
        ).to(device=device, dtype=dtype)
        cds_tower = torch.zeros((batch, 4, 4560), dtype=dtype, device=device)
        cds_tower[:, :, :30] = cds_prefix
        cds_tower[:, :, 30 : 30 + len(self.model.cds)] = cds.expand(batch, -1, -1)
        cds_tower[:, :, 30 + len(self.model.cds) : 60 + len(self.model.cds)] = cds_suffix.expand(
            batch, -1, -1
        )
        return utr5, utr3, cds_tower

    def score_soft(self, probabilities):
        probabilities = probabilities.to(self.device)
        fixed_tail = encode_acgt_sequences(
            [FIXED_KOZAK_TAIL], exact_length=len(FIXED_KOZAK_TAIL)
        ).transpose(1, 2).to(device=probabilities.device, dtype=probabilities.dtype)
        full_probabilities = torch.cat(
            [probabilities, fixed_tail.expand(probabilities.shape[0], -1, -1)], dim=1
        )
        utr5, utr3, cds = self._soft_towers(full_probabilities)
        sequence_features = self.model.model.encode_sequences(utr5, utr3, cds)
        environments = torch.cat(
            [self.model.environment_normal, self.model.environment_treatment], dim=0
        ).to(self.device)
        normal_count = self.model.environment_normal.shape[0]
        sample_count = environments.shape[0]
        batch = probabilities.shape[0]
        flat_environments = environments.unsqueeze(0).expand(batch, -1, -1, -1)
        flat_environments = flat_environments.reshape(
            batch * sample_count, environments.shape[1], 1
        )
        species = torch.full(
            (batch * sample_count, 1),
            self.model.species_id,
            dtype=torch.long,
            device=self.device,
        )
        output = self.model.model.forward_from_sequence_features(
            sequence_features.repeat_interleave(sample_count, dim=0),
            flat_environments,
            species,
        )
        output = output.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        log2_te = 2.0 * torch.log2(torch.tan(torch.arccos(-output) / 2.0))
        log2_te = log2_te.reshape(batch, sample_count)
        normal = log2_te[:, :normal_count].mean(dim=1)
        treatment = log2_te[:, normal_count:].mean(dim=1)
        return torch.stack([treatment - normal, normal, treatment], dim=1), None


def main() -> None:
    cds = read_fasta(Path("/home/hexin/Gluc.fasta"))
    utr3 = read_fasta(Path("/home/hexin/utr3.fasta"))
    initial = make_initial()
    model = load_model(
        "deepadapt_heatshock",
        design_region="utr5",
        cds=cds,
        utr3=utr3,
        weights_dir=WEIGHTS_DIR,
        device="cpu",
    )
    scorer = DeepAdaptDifferentiableScorer(model, utr5_length=len(REFERENCE_UTR5))
    reference_values = scorer.score([REFERENCE_UTR5[:OPTIMIZED_PREFIX_LENGTH]])[0]
    reference_prediction = {
        "sequence": REFERENCE_UTR5,
        "delta_log2_te": float(reference_values[0]),
        "normal_log2_te": float(reference_values[1]),
        "heatshock_log2_te": float(reference_values[2]),
    }
    (WORK_DIR / "reference_prediction.json").write_text(
        json.dumps(reference_prediction, indent=2) + "\n"
    )
    (WORK_DIR / "initial_sequences.json").write_text(
        json.dumps(
            {
                "reference_utr5": REFERENCE_UTR5,
                "fixed_tail": FIXED_KOZAK_TAIL,
                "cds_length": len(cds),
                "utr3_length": len(utr3),
                "initial": [prefix + FIXED_KOZAK_TAIL for prefix in initial],
                "constraints": [function.__name__ for function in CONSTRAINTS],
                "objective": OBJECTIVE,
                "saturation": {
                    "fraction": SATURATION_FRACTION,
                    "minimum_queries": SATURATION.minimum_queries,
                    "plateau_patience_queries": SATURATION.plateau_patience_queries,
                    "maximum_queries": SATURATION.maximum_queries,
                    "material_improvement": SATURATION.material_improvement,
                },
            },
            indent=2,
        )
        + "\n"
    )

    summary = {}
    selected_frames = []
    for algorithm in available_algorithms():
        print(f"[{algorithm}] starting", flush=True)
        try:
            saturated = _optimize_with_saturation(
                scorer,
                OPTIMIZED_PREFIX_LENGTH,
                algorithm=algorithm,
                num_candidates=NUM_CANDIDATES,
                diversity=DIVERSITY,
                initial=initial,
                objective=OBJECTIVE,
                constraint=OPTIMIZER_CONSTRAINTS,
                saturation=SATURATION,
                seed=20260915,
                mutation_rate=0.1,
                elite_fraction=0.5,
                generator_pretrain_epochs=2,
                generator_update_epochs=1,
                den_seed_pretrain_epochs=2,
            )
            result = saturated.at(SATURATION_FRACTION)
            frame = result.to_frame()
            frame["sequence"] = frame["sequence"] + FIXED_KOZAK_TAIL
            frame.insert(0, "stop_reason", saturated.stop_reason)
            frame.insert(0, "total_query_count", saturated.total_query_count)
            frame.insert(0, "selection_query_count", result.query_count)
            frame.insert(0, "saturation_fraction", SATURATION_FRACTION)
            frame.to_csv(WORK_DIR / f"{algorithm}.csv", index=False)
            combined_frame = frame.copy()
            combined_frame.insert(0, "algorithm", algorithm)
            selected_frames.append(combined_frame)
            metadata = {
                "status": "ok",
                "algorithm": algorithm,
                "selection_fraction": SATURATION_FRACTION,
                "selection_query_count": result.query_count,
                "total_query_count": saturated.total_query_count,
                "knee_query_count": saturated.knee_query_count,
                "stop_reason": saturated.stop_reason,
                "sequences": [
                    str(prefix) + FIXED_KOZAK_TAIL for prefix in result.sequences
                ],
                "scores": result.scores.tolist(),
                "metadata": result.metadata,
                "trajectory": saturated.trajectory,
            }
            summary[algorithm] = metadata
            print(
                f"[{algorithm}] done: q90={result.query_count}, "
                f"total={saturated.total_query_count}, stop={saturated.stop_reason}",
                flush=True,
            )
        except Exception as exc:  # Keep the all-algorithm run auditable.
            summary[algorithm] = {
                "status": "error",
                "algorithm": algorithm,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"[{algorithm}] ERROR: {type(exc).__name__}: {exc}", flush=True)
    (WORK_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if selected_frames:
        selected = pd.concat(selected_frames, ignore_index=True)
        selected.to_csv(WORK_DIR / "saturation_0.9_candidates.csv", index=False)
        selected.sort_values("score", ascending=False).groupby(
            "algorithm", sort=False, as_index=False
        ).first().sort_values("score", ascending=False).to_csv(
            WORK_DIR / "best_by_algorithm.csv", index=False
        )
        jointly_improved = selected[
            (selected["activity_delta_log2_te"] >= reference_values[0])
            & (selected["activity_normal_log2_te"] <= reference_values[1])
            & (selected["activity_heatshock_log2_te"] >= reference_values[2])
        ]
        jointly_improved.sort_values("score", ascending=False).to_csv(
            WORK_DIR / "pareto_candidates.csv", index=False
        )


if __name__ == "__main__":
    main()
