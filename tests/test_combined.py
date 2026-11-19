from tests.helpers import run_kernel

import numpy as np
import pytest
import torch

from sequence_optimization import (
    CallableScorer,
    CombinedScorer,
    SequenceSpace,
)


def test_combined_order_budget_and_raw_predictions():
    scorer = CombinedScorer(
        [
            CallableScorer(
                lambda values, base=base: np.array([s.count(base) for s in values])
            )
            for base in "GCT"
        ],
        names=["a", "b", "c"],
    )
    result = run_kernel(
        scorer,
        SequenceSpace(8),
        initial=None,
        objective=["maximize", "minimize", "minimize"],
        num_candidates=4,
        algorithm='basesearch', budget=24,
    )
    expected = np.array([[s.count(base) for base in "GCT"] for s in result.sequences])
    np.testing.assert_array_equal(result.activities, expected)
    np.testing.assert_allclose(result.scores, np.mean(expected * [1, -1, -1], axis=1))
    assert scorer.model_query_counts == dict.fromkeys(
        ["a", "b", "c"], result.query_count
    )


def test_combined_gradients_reach_all_models():
    class Model:
        def __init__(self, column):
            self.column = column

        def score(self, sequences):
            return np.zeros(len(sequences))

        def score_soft(self, values):
            return values[:, :, self.column].sum(dim=1), None

    scorer = CombinedScorer([Model(0), Model(1), Model(2)])
    values = torch.ones((2, 5, 4), requires_grad=True)
    scores, uncertainty = scorer.score_soft(values)
    (scores * torch.tensor([1.0, -1.0, -1.0])).sum().backward()
    np.testing.assert_array_equal(values.grad[0, 0], [1, -1, -1, 0])
    assert uncertainty is None
    assert list(scorer.model_soft_query_counts.values()) == [2, 2, 2]


def test_missing_gradient_support_and_invalid_shapes():
    scorer = CombinedScorer([CallableScorer(lambda values: np.ones((len(values), 2)))])
    with pytest.raises(ValueError, match=r"\[N, 1\]"):
        scorer.score(["AC"])
    with pytest.raises(TypeError, match="every model"):
        scorer.score_soft(torch.zeros(1, 2, 4))
    with pytest.raises(ValueError):
        CombinedScorer([])
    with pytest.raises(ValueError):
        CombinedScorer([scorer, scorer], names=["same", "same"])


def test_partial_uncertainty_is_not_treated_as_zero():
    known = CallableScorer(
        lambda values: (np.ones(len(values)), np.ones(len(values)) * 2)
    )
    unknown = CallableScorer(lambda values: np.ones(len(values)))
    assert CombinedScorer([known, unknown]).score(["A"]).uncertainties is None
    np.testing.assert_array_equal(
        CombinedScorer([known, known]).score(["A"]).uncertainties, [[2, 2]]
    )
