import numpy as np
import pytest
import torch

from sequence_optimization import (
    CallableScorer,
    MultiTaskObjective,
    SequenceSpace,
    basesearch_optimize,
)
from sequence_optimization.objectives import ObjectiveScorer


@pytest.mark.parametrize("aggregation", ["worst", "mean", "smoothmin"])
def test_numpy_torch_values_and_gradients(aggregation):
    objective = MultiTaskObjective(
        high=[0],
        low=[1],
        lower_bounds={0: 1},
        upper_bounds={1: 2},
        scales=[2, 4],
        aggregation=aggregation,
    )
    values = np.array([[5.0, 1.0], [0.0, 4.0]])
    tensor = torch.tensor(values, requires_grad=True)
    soft = objective.scalarize_soft(tensor)
    np.testing.assert_allclose(soft.detach(), objective.scalarize(values))
    soft.sum().backward()
    assert torch.isfinite(tensor.grad).all()
    assert (tensor.grad[:, 0] >= 0).all()
    assert (tensor.grad[:, 1] <= 0).all()
    eps = 1e-5
    for i in range(2):
        delta = np.zeros_like(values)
        delta[:, i] = eps
        derivative = (
            objective.scalarize(values + delta) - objective.scalarize(values - delta)
        ) / (2 * eps)
        np.testing.assert_allclose(tensor.grad[:, i], derivative, atol=1e-6)


def test_worst_and_threshold_semantics():
    values = np.array([[10.0, -1.0], [2.0, 2.0]])
    worst = MultiTaskObjective(high="all")
    assert worst.scalarize(values)[1] > worst.scalarize(values)[0]
    specific = MultiTaskObjective(high=[0], low=[1])
    assert specific.scalarize(values)[0] > specific.scalarize(values)[1]
    bounded = MultiTaskObjective(
        high=[0], low=[1], lower_bounds={0: 3}, upper_bounds={1: 0}
    )
    assert bounded.evaluate(values)["all_satisfied"] == [True, False]
    assert worst.evaluate(values)["all_satisfied"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"high": [0], "low": [0]},
        {"high": [0, 0]},
        {"high": [3]},
        {"high": [True]},
        {"high": ["missing"]},
        {"high": "all", "scales": [1, 0]},
        {"high": "all", "scales": [1]},
        {"high": "all", "centers": [0, float("nan")]},
        {"high": "all", "lower_bounds": {0: 2}},
        {"high": [0], "upper_bounds": {1: 1}},
        {"high": [0], "lower_bounds": {0: float("inf")}},
    ],
)
def test_invalid_task_configuration(kwargs):
    with pytest.raises(ValueError):
        MultiTaskObjective(**kwargs).scalarize(np.zeros((2, 2)), ("a", "b"))


def test_result_contains_raw_activities_and_threshold_details():
    scorer = CallableScorer(
        lambda seqs: np.array([[s.count("G"), s.count("C")] for s in seqs]),
        task_names=("a", "b"),
    )
    objective = MultiTaskObjective(
        high=["a"], low=["b"], lower_bounds={"a": 2}, upper_bounds={"b": 1}
    )
    result = basesearch_optimize(
        scorer, SequenceSpace(8), budget=24, num_candidates=4, objective=objective
    )
    np.testing.assert_allclose(
        result.scores, objective.scalarize(result.activities, result.task_names)
    )
    frame = result.to_frame()
    assert list(frame.all_satisfied) == list(
        (frame.activity_a >= 2) & (frame.activity_b <= 1)
    )
    assert "worst_margin" in frame


def test_multitask_uncertainty():
    scorer = CallableScorer(
        lambda seqs: (np.tile([2.0, 4.0], (len(seqs), 1)), np.ones((len(seqs), 2)))
    )
    wrapped = ObjectiveScorer(
        scorer, MultiTaskObjective(high="all", aggregation="mean")
    )
    np.testing.assert_allclose(wrapped.score(["AC"]).uncertainties, [np.sqrt(0.5)])


@pytest.mark.parametrize("aggregation", ["worst", "mean", "smoothmin"])
def test_soft_uncertainty_matches_numpy(aggregation):
    class Scorer:
        def score_soft(self, values):
            return values, values * 0.1

    values = torch.tensor([[2.0, 4.0]], requires_grad=True)
    objective = MultiTaskObjective(high="all", aggregation=aggregation)
    utility, uncertainty = ObjectiveScorer(Scorer(), objective).score_soft(values)
    np.testing.assert_allclose(
        uncertainty.detach(),
        objective.uncertainty(values.detach().numpy(), values.detach().numpy() * 0.1),
        rtol=1e-6,
    )
    (utility + uncertainty).sum().backward()
    assert torch.isfinite(values.grad).all()
