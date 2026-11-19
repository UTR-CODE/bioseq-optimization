from tests.helpers import run_kernel

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
        result = run_kernel(
            model,
            SequenceSpace(8),
            initial=["AAAAAAAA", "CCCCCCCC"],
            objective=["maximize"],
            num_candidates=2,
            algorithm='basesearch', budget=50,
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
    result = run_kernel(
        CallableScorer(lambda values: np.zeros(len(values))),
        SequenceSpace(1, "AC"),
        initial=["A"],
        objective=["maximize"],
        num_candidates=1,
        algorithm='basesearch', budget=10,
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
        ("basesearch_batch_size", 0),
        ("cross_entropy_batch_size", 0),
    ],
)
def test_invalid_config(key, value):
    with pytest.raises(ValueError, match=key):
        OptimizationConfig(4, num_candidates=4, **{key: value})


def test_initial_only_budget_and_saturation():
    scorer = CallableScorer(lambda values: np.zeros(len(values)))
    result = run_kernel(
        scorer, SequenceSpace(8), None, ["maximize"], 4, algorithm='basesearch', budget=4
    )
    assert result.query_count == 4
    assert result.trajectory == []
    saturated = basesearch_optimize(
        scorer,
        SequenceSpace(8),
        None,
        ["maximize"],
        4,
        saturation=SaturationConfig(
            minimum_queries=8, plateau_patience_queries=8
        ),
    )
    assert saturated.stop_reason == "plateau"
    assert saturated.total_query_count >= 8
    assert saturated.selected.scores.tolist() == [0.] * 4


def test_basesearch_scores_mutations_in_batches():
    class BatchModel:
        task_names = ("activity",)

        def __init__(self):
            self.batch_sizes = []

        def score(self, sequences):
            self.batch_sizes.append(len(sequences))
            return np.asarray([[sequence.count("G")] for sequence in sequences])

    model = BatchModel()
    initial = ["AAAAAA", "CCCCCC", "ACACAC", "CACACA"]
    result = run_kernel(
        model,
        length=6,
        initial=initial,
        objective=["maximize"],
        num_candidates=2,
        algorithm='basesearch', budget=20,
        diversity=0,
        basesearch_batch_size=8,
        seed=7,
    )
    assert model.batch_sizes[0] == len(initial)
    assert max(model.batch_sizes[1:]) > 1
    assert result.metadata["batch_size"] == 8
