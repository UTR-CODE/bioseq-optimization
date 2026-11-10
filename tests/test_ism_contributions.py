import numpy as np
import pytest

from sequence_optimization.attribution import ism_contributions


def test_exact_centered_effects_on_position_weighted_additive_model():
    weights = np.array(
        [[0.0, 1.0, 2.0, 3.0], [4.0, -1.0, 2.0, 0.0], [-3.0, 0.0, 1.0, 2.0]]
    )
    queried = []

    def score(sequences):
        queried.extend(sequences)
        return np.array(
            [
                sum(weights[i, "ACGT".index(b)] for i, b in enumerate(s))
                for s in sequences
            ]
        )

    result = ism_contributions(["ACG", "TTA"], score, batch_size=5)
    for row, seq in enumerate(result.sequences):
        expected = np.array(
            [weights[i, "ACGT".index(b)] - weights[i].mean() for i, b in enumerate(seq)]
        )
        np.testing.assert_allclose(result.observed[row], expected)
        np.testing.assert_allclose(result.logo[row].sum(axis=1), expected)
        for i, base in enumerate(seq):
            assert (
                result.mutant_predictions[row, i, "ACGT".index(base)]
                == result.reference_predictions[row]
            )
            assert (
                np.count_nonzero(result.logo[row, i, np.array(list("ACGT")) != base])
                == 0
            )
    assert len(queried) == 2 * (1 + 3 * 3)


def test_context_average_preserves_fixed_flanks_for_each_mutant():
    contexts = [("AA", "CC"), ("GG", "TT")]
    constructs = []

    def score(sequences):
        batch = [left + s + right for s in sequences for left, right in contexts]
        constructs.extend(batch)
        return (
            np.array([s.count("G") for s in batch]).reshape(len(sequences), 2).mean(1)
        )

    result = ism_contributions(["GCAA"], score)
    np.testing.assert_allclose(result.observed[0], [0.75, -0.25, -0.25, -0.25])
    assert all(s[:2] + s[-2:] in {"AACC", "GGTT"} for s in constructs)


def test_nonadditive_interactions_not_claimed_as_output_decomposition():
    def score(sequences):
        return np.array([float(s == "GG") for s in sequences])
    result = ism_contributions(["GG"], score)
    np.testing.assert_allclose(result.observed, [[0.75, 0.75]])
    assert result.observed.sum() != result.reference_predictions.sum()


def test_bad_score_shape_rejected():
    with pytest.raises(ValueError, match="finite"):
        ism_contributions(["AAAA"], lambda s: np.zeros((len(s), 2)))
