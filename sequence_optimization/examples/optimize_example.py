"""Small standalone example; replace the scorer with any user model."""

import numpy as np

from sequence_optimization import CallableScorer, OptimizationConfig, SequenceSpace, run


def gc_content(sequences: np.ndarray) -> np.ndarray:
    return np.asarray([sum(base in "GC" for base in sequence) for sequence in sequences], dtype=float)


if __name__ == "__main__":
    space = SequenceSpace(length=20, alphabet="ACGT")
    scorer = CallableScorer(gc_content)
    config = OptimizationConfig(budget=10_000, num_candidates=100)
    result = run("genetic_algorithm", scorer, space, config, seed=7)
    print(result.to_frame().head())
