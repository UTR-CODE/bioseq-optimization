import numpy as np
import pytest

from sequence_optimization import (
    OptimizationConfig,
    SequenceSpace,
    basesearch_optimize,
    CallableScorer,
    SaturationConfig,
)
from sequence_optimization.advanced import _decay_base_priority


@pytest.mark.parametrize("offset", [0, -100])
def test_search_improves_and_preserves_original_scores(offset):
    class Model:
        task_names = ("activity",)

        def __init__(self):
            self.queried = []

        def score(self, sequences):
            self.queried.extend(sequences)
            return np.asarray([[s.count("G") + offset] for s in sequences])

    results = []
    for _ in range(2):
        model = Model()
        result = basesearch_optimize(
            model,
            SequenceSpace(8),
            initial=["AAAAAAAA", "CCCCCCCC"],
            num_candidates=2,
            budget=50,
            seed=7,
            basesearch_queue_size=3,
        )
        assert result.query_count == len(model.queried) == len(set(model.queried)) == 50
        assert result.scores[0] > offset
        assert result.metadata["pruned"] > 0
        assert result.metadata["queue_size"] <= 6
        np.testing.assert_array_equal(
            result.scores, [s.count("G") + offset for s in result.sequences]
        )
        np.testing.assert_array_equal(result.activities[:, 0], result.scores)
        results.append(result)
    np.testing.assert_array_equal(results[0].sequences, results[1].sequences)


def test_tiny_space_exhaustion_and_duplicate_budget():
    result = basesearch_optimize(
        CallableScorer(lambda values: np.zeros(len(values))),
        SequenceSpace(1, "AC"),
        initial=["A"],
        num_candidates=1,
        budget=10,
    )
    assert result.query_count == 2
    assert result.metadata["stop_reason"] == "space_exhausted"


@pytest.mark.parametrize("score", [-10.0, 0.0, 10.0])
def test_priority_decay_demotes_all_score_signs(score):
    assert _decay_base_priority(score, 0.995) < score
    if score > 0:
        assert _decay_base_priority(score, 0.995) == pytest.approx(score * 0.995)


@pytest.mark.parametrize(
    "key,value",
    [
        ("basesearch_decay", 1),
        ("basesearch_decay", float("nan")),
        ("basesearch_queue_size", 0),
        ("basesearch_queue_size", 1.5),
        ("basesearch_max_stalled", 0),
    ],
)
def test_invalid_config(key, value):
    with pytest.raises(ValueError, match=key):
        OptimizationConfig(4, num_candidates=4, **{key: value})


def test_initial_only_budget_and_saturation():
    scorer = CallableScorer(lambda values: np.zeros(len(values)))
    result = basesearch_optimize(scorer, SequenceSpace(8), num_candidates=4, budget=4)
    assert result.query_count == 4
    assert result.trajectory == []
    saturated = basesearch_optimize(
        scorer,
        SequenceSpace(8),
        num_candidates=4,
        saturation=SaturationConfig(
            minimum_queries=8, plateau_patience_queries=8, maximum_queries=40
        ),
    )
    assert saturated.total_query_count <= 40
