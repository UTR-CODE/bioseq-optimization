import numpy as np
import pytest
import torch

from sequence_optimization import CallableScorer, MultiTaskObjective, generate_designs
from sequence_optimization.objectives import ObjectiveScorer
from sequence_optimization.algorithms import available_algorithms, optimize


@pytest.mark.parametrize("algorithm", available_algorithms())
def test_all_optimizers_share_directional_stop(algorithm):
    class Model:
        def score(self, seqs):
            return np.array([[s.count("G"), s.count("C")] for s in seqs], dtype=float)

        def score_soft(self, probabilities):
            return torch.stack(
                [probabilities[:, :, 2].sum(1), probabilities[:, :, 1].sum(1)], dim=1
            ), None

    result = optimize(
        Model(),
        length=4,
        initial=["AAAA", "GGGG"],
        objective=["maximize", "minimize"],
        stop_values=[4, 0],
        num_candidates=2,
        algorithm=algorithm,
        budget=20,
        diversity=0,
    )
    assert result.metadata["stop_reason"] == "target_reached"
    assert result.query_count == 2
    assert result.metadata["objective"]["stop_values"] == [4, 0]


def test_minimize_line_gates_then_jointly_stops():
    values = {"AAAA": [0, 0, 999], "CCCC": [0, -2, -999], "GGGG": [6, -2, 0]}
    scorer = CallableScorer(lambda seqs: np.array([values[s] for s in seqs]))
    wrapped = ObjectiveScorer(
        scorer, ["maximize", "minimize", "ignore"], stop_values=[6, -2, None]
    )
    wrapped.score(["AAAA", "CCCC"])
    assert wrapped.thresholds_passed(["AAAA", "CCCC"]).tolist() == [False, True]
    assert not wrapped.target_stop_reached
    wrapped.score(["GGGG"])
    assert wrapped.target_stop_reached
    assert wrapped.thresholds_passed(["AAAA", "CCCC", "GGGG"]).tolist() == [
        False,
        False,
        True,
    ]


@pytest.mark.parametrize("aggregation", ["mean", "worst", "smoothmin"])
def test_stop_values_numpy_torch_and_minimize_gradient(aggregation):
    obj = MultiTaskObjective(
        directions=["maximize", "minimize", "ignore"],
        stop_values=[6, -2, None],
        aggregation=aggregation,
    )
    x = torch.tensor([[2.0, 1.0, 90.0], [5.0, -1.0, -90.0]], requires_grad=True)
    y = obj.scalarize_soft(x)
    np.testing.assert_allclose(y.detach(), obj.scalarize(x.detach().numpy()), atol=1e-6)
    y.sum().backward()
    assert (x.grad[:, 0] >= 0).all()
    assert (x.grad[:, 1] <= 0).all()
    assert (x.grad[:, 2] == 0).all()


@pytest.mark.parametrize(
    "directions, stops",
    [
        (["maximize"], [True]),
        (["maximize"], [float("inf")]),
        (["maximize"], []),
        (["maximize", "ignore"], [None, 2]),
        ([6.0], [None]),
        (["minimize"], [float("nan")]),
    ],
)
def test_invalid_directions_and_lines(directions, stops):
    with pytest.raises(ValueError):
        MultiTaskObjective(directions=directions, stop_values=stops)


def test_default_lines_unlimited():
    obj = MultiTaskObjective(directions=["maximize", "minimize", "ignore"])
    assert obj.stop_values is None
    assert obj.target_thresholds(3) == {}
    assert obj.has_open_objectives()


def test_suite_exposes_minimize_line():
    frame = generate_designs(
        CallableScorer(lambda seqs: np.array([s.count("G") for s in seqs])),
        length=4,
        initial=["AAAA", "GGGG"],
        objective=["minimize"],
        stop_values=[0],
        algorithms=["random", "genetic_algorithm"],
        num_candidates=2,
        diversity=0,
        budget=20,
    )
    assert len(frame) == 4
    assert set(frame.algorithm) == {"random", "genetic_algorithm"}
