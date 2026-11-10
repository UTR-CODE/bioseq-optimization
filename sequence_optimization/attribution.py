"""Exact single-nucleotide contributions for scalar sequence scorers."""

from dataclasses import dataclass

import numpy as np

from .api import SequenceSpace, normalize_nucleotide_sequences


@dataclass
class ISMContributions:
    sequences: np.ndarray
    reference_predictions: np.ndarray
    mutant_predictions: np.ndarray
    observed: np.ndarray
    logo: np.ndarray


def ism_contributions(sequences, score_fn, *, batch_size=256):
    """Score all 3*L substitutions and center four-base predictions per position.

    ``score_fn`` must return one finite scalar per sequence. For the actual base
    at position i, contribution = f(sequence) - mean_b f(sequence[i <- b]).
    The original-base prediction is reused, not scored as another mutation.
    ``logo`` masks centered scores to the actual base, in A/C/G/T channel order.
    These local substitution effects are neither additive output decompositions
    nor motif interaction measurements. They require no gradient approximation.
    """
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, (int, np.integer))
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    values = normalize_nucleotide_sequences(sequences)
    if not len(values):
        raise ValueError("at least one sequence is required")
    space = SequenceSpace(len(values[0]))
    codes = space.encode(values)

    def score(batch):
        result = np.asarray(score_fn(np.asarray(batch, dtype=str)), dtype=float)
        if result.shape == (len(batch), 1):
            result = result[:, 0]
        if result.shape != (len(batch),) or not np.isfinite(result).all():
            raise ValueError(
                "score_fn must return a finite [N] scalar prediction vector"
            )
        return result

    reference = np.concatenate(
        [score(values[i : i + batch_size]) for i in range(0, len(values), batch_size)]
    )
    mutants = np.empty((len(values), space.length, 4), dtype=float)
    for row, sequence in enumerate(values):
        mutants[row, np.arange(space.length), codes[row]] = reference[row]
        pending, indices = [], []
        for position, actual in enumerate(sequence):
            for base_index, base in enumerate("ACGT"):
                if base == actual:
                    continue
                pending.append(sequence[:position] + base + sequence[position + 1 :])
                indices.append((position, base_index))
                if len(pending) == batch_size:
                    predictions = score(pending)
                    for index, prediction in zip(indices, predictions):
                        mutants[(row, *index)] = prediction
                    pending, indices = [], []
        if pending:
            for index, prediction in zip(indices, score(pending)):
                mutants[(row, *index)] = prediction
    observed = reference[:, None] - mutants.mean(axis=2)
    logo = np.eye(4)[codes] * observed[:, :, None]
    return ISMContributions(values, reference, mutants, observed, logo)
