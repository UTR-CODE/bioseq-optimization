import numpy as np
import pytest
import torch

from sequence_optimization import (
    CallableScorer,
    MultiTaskObjective,
    SaturationConfig,
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
        Scorer(), SequenceSpace(8), objective=directions, budget=24, num_candidates=4
    )
    np.testing.assert_allclose(
        result.scores,
        (result.activities[:, 0] - result.activities[:, 1]) / 2,
    )
    assert result.metadata["objective"]["aggregation"] == "mean"
    assert result.metadata["objective"]["model_weights"] == [1.0, 1.0]
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


def test_direction_list_accepts_normalized_model_weights():
    scorer = CallableScorer(
        lambda sequences: np.array(
            [[sequence.count("G"), sequence.count("C")] for sequence in sequences]
        )
    )
    wrapped = ObjectiveScorer(
        scorer,
        ["maximize", "minimize"],
        model_weights=[2.0, 1.0],
    )
    batch = wrapped.score(["GGCC", "GCCC"])
    np.testing.assert_allclose(batch.scores, [(2 * 2 - 2) / 3, (2 * 1 - 3) / 3])
    assert wrapped.objective.describe(2)["model_weights"] == [2.0, 1.0]

    ignored = MultiTaskObjective(
        directions=["maximize", "minimize", "ignore"],
        model_weights=[2.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(ignored.scalarize([[2.0, 2.0, 1e9]]), [2 / 3])

    result = basesearch_optimize(
        scorer,
        SequenceSpace(8),
        objective=["maximize", "minimize"],
        model_weights=[2.0, 1.0],
        budget=12,
        num_candidates=4,
    )
    np.testing.assert_allclose(
        result.scores,
        (2 * result.activities[:, 0] - result.activities[:, 1]) / 3,
    )


def test_direction_list_weights_apply_to_soft_scores_and_gradients():
    class Scorer:
        def score_soft(self, values):
            return values, None

    values = torch.tensor([[2.0, 4.0]], requires_grad=True)
    wrapped = ObjectiveScorer(
        Scorer(),
        ["maximize", "minimize"],
        model_weights=[3.0, 1.0],
    )
    score, uncertainty = wrapped.score_soft(values)
    np.testing.assert_allclose(score.detach(), [(3 * 2 - 4) / 4])
    assert uncertainty is None
    score.sum().backward()
    np.testing.assert_allclose(values.grad, [[0.75, -0.25]])


@pytest.mark.parametrize(
    "model_weights",
    ([1.0], [1.0, -1.0], [0.0, 1.0], [float("nan"), 1.0]),
)
def test_invalid_model_weights(model_weights):
    scorer = CallableScorer(lambda values: np.zeros((len(values), 2)))
    with pytest.raises(ValueError):
        basesearch_optimize(
            scorer,
            SequenceSpace(8),
            objective=["maximize", "maximize"],
            model_weights=model_weights,
            budget=4,
            num_candidates=4,
        )


def test_model_weights_require_a_direction_list():
    scorer = CallableScorer(lambda values: np.zeros((len(values), 2)))
    with pytest.raises(ValueError, match="direction list"):
        basesearch_optimize(
            scorer,
            SequenceSpace(8),
            objective=MultiTaskObjective(high="all"),
            model_weights=[1.0, 1.0],
            budget=4,
            num_candidates=4,
        )


def test_model_weights_apply_during_saturation():
    scorer = CallableScorer(
        lambda values: np.array(
            [[value.count("G"), value.count("C")] for value in values]
        )
    )
    saturated = genetic_algorithm_optimize(
        scorer,
        SequenceSpace(4),
        objective=["maximize", "minimize"],
        model_weights=[3.0, 1.0],
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        num_candidates=4,
        saturation=SaturationConfig(
            fractions=(0.8,),
            minimum_queries=4,
            plateau_patience_queries=4,
            maximum_queries=12,
        ),
        seed=3,
    )
    selected = saturated[0.8]
    np.testing.assert_allclose(
        selected.scores,
        (3 * selected.activities[:, 0] - selected.activities[:, 1]) / 4,
    )
    assert selected.metadata["objective"]["model_weights"] == [3.0, 1.0]


@pytest.mark.parametrize(
    "directions", [[], ["ignore", "ignore"], ["up", "down"], ["maximize"]]
)
def test_invalid_direction_lists(directions):
    scorer = CallableScorer(lambda values: np.zeros((len(values), 2)))
    with pytest.raises(ValueError):
        basesearch_optimize(
            scorer, SequenceSpace(8), objective=directions, budget=4, num_candidates=4
        )


def test_single_output_list():
    scorer = CallableScorer(lambda values: np.array([s.count("G") for s in values]))
    result = basesearch_optimize(
        scorer, SequenceSpace(8), objective=["maximize"], budget=12, num_candidates=4
    )
    np.testing.assert_array_equal(result.scores, result.activities[:, 0])


@pytest.mark.parametrize("aggregation", ["worst", "mean", "smoothmin"])
def test_numpy_torch_values_and_gradients(aggregation):
    objective = MultiTaskObjective(
        high=[0],
        low=[1],
        lower_bounds={0: 1},
        upper_bounds={1: 2},
        scales=[2, 4],
        model_weights=[2, 0.5],
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
        {"high": "all", "model_weights": [1, 0]},
        {"high": "all", "model_weights": [1]},
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
