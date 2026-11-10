import numpy as np
import pytest
import torch

from sequence_optimization import (
    CachedScorer,
    CallableScorer,
    MultiTaskObjective,
    SequenceSpace,
    basesearch_optimize,
    genetic_algorithm_optimize,
)
from sequence_optimization.objectives import ObjectiveScorer


def test_direction_list_is_model_independent_and_ignores_third_output():
    class Scorer:
        def score(self, sequences):
            return np.array(
                [[s.count("G"), s.count("C"), len(s) * 1000] for s in sequences]
            )

        def score_soft(self, probabilities):
            return probabilities.sum(dim=1)[:, :3], None

    directions = ["maximize", "minimize", "ignore"]
    result = basesearch_optimize(
        Scorer(), SequenceSpace(8), None, directions, 4, budget=24
    )
    np.testing.assert_allclose(
        result.scores,
        (result.activities[:, 0] - result.activities[:, 1]) / 2,
    )
    assert result.metadata["objective"]["aggregation"] == "mean"
    assert "model_weights" not in result.metadata["objective"]
    assert result.activities.shape == (4, 3)
    objective = MultiTaskObjective(directions=directions)
    values = torch.tensor([[1.0, -2.0, 1e6], [3.0, -1.0, -1e6]], requires_grad=True)
    objective.scalarize_soft(values).sum().backward()
    np.testing.assert_array_equal(values.grad[:, 2], [0, 0])
    changed = values.detach().numpy().copy()
    changed[:, 2] *= -100
    np.testing.assert_array_equal(
        objective.scalarize(changed), objective.scalarize(values.detach().numpy())
    )


def test_direction_list_uses_equal_weights():
    scorer = CallableScorer(
        lambda sequences: np.array(
            [[sequence.count("G"), sequence.count("C")] for sequence in sequences]
        )
    )
    wrapped = ObjectiveScorer(scorer, ["maximize", "minimize"])
    batch = wrapped.score(["GGCC", "GCCC"])
    np.testing.assert_allclose(batch.scores, [0.0, -1.0])
    assert "model_weights" not in wrapped.objective.describe(2)

    ignored = MultiTaskObjective(directions=["maximize", "minimize", "ignore"])
    np.testing.assert_allclose(ignored.scalarize([[2.0, 2.0, 1e9]]), [0.0])

    result = basesearch_optimize(
        scorer,
        SequenceSpace(8),
        initial=None,
        objective=["maximize", "minimize"],
        budget=12,
        num_candidates=4,
    )
    np.testing.assert_allclose(
        result.scores,
        (result.activities[:, 0] - result.activities[:, 1]) / 2,
    )


def test_direction_list_equal_weights_apply_to_soft_scores_and_gradients():
    class Scorer:
        def score_soft(self, values):
            return values, None

    values = torch.tensor([[2.0, 4.0]], requires_grad=True)
    wrapped = ObjectiveScorer(Scorer(), ["maximize", "minimize"])
    score, uncertainty = wrapped.score_soft(values)
    np.testing.assert_allclose(score.detach(), [-1.0])
    assert uncertainty is None
    score.sum().backward()
    np.testing.assert_allclose(values.grad, [[0.5, -0.5]])


@pytest.mark.parametrize(
    "directions", [[], ["ignore", "ignore"], ["up", "down"], ["maximize"]]
)
def test_invalid_direction_lists(directions):
    scorer = CallableScorer(lambda values: np.zeros((len(values), 2)))
    with pytest.raises(ValueError):
        basesearch_optimize(scorer, SequenceSpace(8), None, directions, 4, budget=4)


def test_single_output_list():
    scorer = CallableScorer(lambda values: np.array([s.count("G") for s in values]))
    result = basesearch_optimize(
        scorer,
        SequenceSpace(8),
        initial=None,
        objective=["maximize"],
        budget=12,
        num_candidates=4,
        diversity=0,
    )
    np.testing.assert_array_equal(result.scores, result.activities[:, 0])


def test_numeric_objective_bound_continues_and_is_reported():
    scorer = CallableScorer(
        lambda values: np.asarray([value.count("G") for value in values]),
        task_names=("activity",),
    )
    result = genetic_algorithm_optimize(
        scorer,
        SequenceSpace(4),
        objective=["maximize"],
        stop_values=[4.0],
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        budget=40,
        num_candidates=4,
        diversity=0,
        seed=3,
    )
    assert result.query_count > 4
    assert result.metadata["stop_reason"] != "target_reached"
    assert result.metadata["objective"]["target_thresholds"] == {"activity": 4.0}
    frame = result.to_frame()
    assert frame.iloc[0].activity_activity == 4
    assert frame.iloc[0].satisfied_activity


def test_numeric_bound_is_stationary_for_all_objectives():
    scorer = CallableScorer(
        lambda values: np.asarray(
            [[value.count("G"), value.count("C")] for value in values]
        )
    )
    wrapped = ObjectiveScorer(scorer, ["maximize", "maximize"], stop_values=[2.0, None], stop_tolerance=None)
    before = wrapped.score(["AAAA", "GCCC", "CCCC"])
    np.testing.assert_allclose(before.scores[:2], [0.0, 2.0])
    wrapped.score(["GGAA", "GGCC"])
    rescored = wrapped.scores_for(["CCCC", "GGAA", "GGCC"])
    np.testing.assert_allclose(rescored, [2, 1, 2])
    assert wrapped.thresholds_passed(["CCCC", "GGAA", "GGCC"]).tolist() == [
        True,
        True,
        True,
    ]
    assert not wrapped.target_stop_reached


def test_numeric_target_rescores_cached_uncertainty():
    scorer = CallableScorer(
        lambda values: (
            np.asarray([[value.count("G"), value.count("C")] for value in values]),
            np.ones((len(values), 2)),
        )
    )
    cached = CachedScorer(
        ObjectiveScorer(scorer, ["maximize", "maximize"], stop_values=[2.0, None], stop_tolerance=None),
        budget=2,
    )
    first = cached.score(["CCCC"])
    np.testing.assert_allclose(first.uncertainties, [np.sqrt(0.5)])
    cached.score(["GGCC"])
    rescored = cached.score(["CCCC"])
    np.testing.assert_allclose(rescored.uncertainties, [np.sqrt(0.5)])


def test_numeric_bounds_reject_each_violating_candidate():
    activities = {"LEFT": [3.0, 0.0], "RIGHT": [0.0, 3.0]}
    scorer = CallableScorer(
        lambda values: np.asarray([activities[str(value)] for value in values])
    )
    wrapped = ObjectiveScorer(scorer, ["maximize", "maximize"], stop_values=[2.0, 2.0])
    wrapped.score(["left", "right"])
    assert wrapped.thresholds_passed(["left", "right"]).tolist() == [False, False]
    assert not wrapped.target_stop_reached


def test_numeric_target_can_return_default_knee_fraction_selection():
    scorer = CallableScorer(lambda values: [value.count("G") for value in values])
    result = genetic_algorithm_optimize(
        scorer,
        SequenceSpace(4),
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        objective=["maximize"],
        stop_values=[4.0],
        num_candidates=4,
        diversity=0,
    )
    assert result.query_count > 4
    assert result.metadata["saturation_fraction"] == 0.9
    assert result.metadata["selection"] == "quality_diversity_near_knee_gain"
    assert result.metadata["stop_reason"] != "target_reached"
    assert (result.activities[:, 0] <= 4).all()
    np.testing.assert_allclose(result.scores, np.minimum(result.activities[:, 0] - 3.5, 0))


@pytest.mark.parametrize(
    "directions",
    [[float("nan")], [float("inf")], [True], ["maximize", object()]],
)
def test_invalid_numeric_objective_targets(directions):
    with pytest.raises(ValueError, match="maximize, minimize, or ignore"):
        MultiTaskObjective(directions=directions)


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
    worst = MultiTaskObjective(high="all", aggregation="worst")
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
        scorer, SequenceSpace(8), None, objective, 4, budget=24
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
