import numpy as np
import pytest
import torch

from sequence_optimization import CallableScorer, MultiTaskObjective, SaturationConfig, generate_designs
from sequence_optimization.objectives import ObjectiveScorer
from sequence_optimization.algorithms import available_algorithms, optimize


@pytest.mark.parametrize("algorithm", available_algorithms())
def test_all_optimizers_share_directional_bounds(algorithm):
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
        stop_values=[4, 0], stop_tolerance=None,
        num_candidates=2,
        algorithm=algorithm,
        budget=32,
        diversity=0,
        generator_pretrain_epochs=1,
        generator_update_epochs=1,
        den_seed_pretrain_epochs=1,
    )
    assert result.metadata["stop_reason"] != "target_reached"
    assert result.query_count >= 2
    assert algorithm == "random" or result.trajectory
    assert (result.activities[:, 0] <= 4).all()
    assert (result.activities[:, 1] >= 0).all()
    assert result.metadata["objective"]["stop_values"] == [4, 0]


def test_bounds_never_deactivate_or_stop_tasks():
    values = {"AAAA": [0, 0, 999], "CCCC": [0, -2, -999], "GGGG": [6, -2, 0],
              "TTTT": [6.01, -2.01, 0]}
    scorer = CallableScorer(lambda seqs: np.array([values[s] for s in seqs]))
    wrapped = ObjectiveScorer(
        scorer, ["maximize", "minimize", "ignore"], stop_values=[6, -2, None], stop_tolerance=None
    )
    wrapped.score(["AAAA", "CCCC"])
    assert wrapped.thresholds_passed(["AAAA", "CCCC"]).tolist() == [True, True]
    assert not wrapped.target_stop_reached
    wrapped.score(["GGGG"])
    assert not wrapped.target_stop_reached
    assert wrapped.thresholds_passed(["AAAA", "CCCC", "GGGG"]).tolist() == [
        True,
        True,
        True,
    ]
    before = wrapped.scores_for(["AAAA", "CCCC", "GGGG"])
    wrapped.score(["TTTT"])
    assert not wrapped.thresholds_passed(["TTTT"])[0]
    np.testing.assert_array_equal(before, wrapped.scores_for(["AAAA", "CCCC", "GGGG"]))
    assert wrapped.objective_version == 0
    assert wrapped._reached_targets == set()


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
        (["maximize"], [(6, -2)]),
        (["maximize"], [(0, 1, 2)]),
        (["maximize"], [(None, float("inf"))]),
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


@pytest.mark.parametrize("algorithm", available_algorithms())
@pytest.mark.parametrize("saturation", [False, True])
def test_interval_bounds_all_algorithms_and_final_selection(algorithm, saturation):
    if saturation and algorithm == "random":
        pytest.skip("random screening has a fixed budget")

    class Model:
        def score(self, seqs):
            return np.asarray([[s.count("G"), s.count("C") - 2, s.count("T") - 2]
                               for s in seqs], dtype=float)

        def score_soft(self, p):
            return torch.stack([p[:, :, 2].sum(1), p[:, :, 1].sum(1) - 2,
                                p[:, :, 3].sum(1) - 2], dim=1), None

    model = Model()
    options = {"saturation": SaturationConfig(
        minimum_queries=16, plateau_patience_queries=32, maximum_queries=100
    )} if saturation else {"budget": 100}
    result = optimize(
        model, 4, ["AAAA", "GGGG"], ["maximize", "minimize", "minimize"], 2,
        stop_values=[(-2, 2)] * 3, stop_tolerance=None, algorithm=algorithm, diversity=0,
        generator_pretrain_epochs=1, generator_update_epochs=1,
        den_seed_pretrain_epochs=1, **options,
    )
    selected = result.selected if hasattr(result, "selected") else result
    assert len(selected.sequences) == 2
    assert np.all((selected.activities >= -2) & (selected.activities <= 2))
    assert "GGGG" not in selected.sequences
    np.testing.assert_allclose(selected.scores, (
        selected.activities[:, 0] - selected.activities[:, 1] - selected.activities[:, 2]
    ) / 3)
    assert all(selected.metadata["objective_evaluation"]["all_satisfied"])
    assert selected.metadata["stop_reason"] != "target_reached"


def test_boundary_inclusive_and_no_cross_candidate_latching():
    obj = MultiTaskObjective(directions=["maximize", "minimize", "minimize"],
                             stop_values=[(-2, 6)] * 3, stop_tolerance=None)
    x = np.array([[6, -2, -2], [5, 0, 0], [6.00001, 0, 0], [1, -2.00001, 0],
                  [-2.00001, 0, 0], [1, 6.00001, 0]])
    assert obj.feasible(x).tolist() == [True, True, False, False, False, False]
    np.testing.assert_allclose(obj.scalarize(x[:2], reached_targets={0, 1, 2}), [10/3, 5/3])


def test_soft_gradient_keeps_all_tasks_and_pushes_overshoot_back():
    class Model:
        def score_soft(self, x):
            return x, None

    wrapped = ObjectiveScorer(Model(), ["maximize", "minimize", "minimize"],
                              stop_values=[(-2, 6)] * 3, stop_tolerance=None)
    x = torch.tensor([[5., 0., 0.], [7., -3., 7.]], requires_grad=True)
    utility, _ = wrapped.score_soft(x)
    utility.sum().backward()
    np.testing.assert_allclose(x.grad[0], [1/3, -1/3, -1/3])
    assert x.grad[1, 0] < 0
    assert x.grad[1, 1] > 0
    assert x.grad[1, 2] < 0


def test_impossible_interval_fails_without_invalid_padding():
    with pytest.raises(ValueError, match="feasible initial"):
        optimize(CallableScorer(lambda seqs: np.full(len(seqs), 7.)), 4,
                 ["AAAA", "CCCC"], ["maximize"], 2, algorithm="random",
                 stop_values=[(-2, 6)], budget=20, diversity=0)


def test_prediction_bounds_with_sequence_constraints():
    result = optimize(CallableScorer(lambda seqs: [s.count("G") for s in seqs]),
                      4, ["AAAA", "CCCC"], ["maximize"], 2, algorithm="random",
                      stop_values=[(0, 2)], budget=80, diversity=0,
                      constraint=lambda seqs: np.array(["T" not in s for s in seqs], dtype=float))
    assert all("T" not in s for s in result.sequences)
    assert np.all(result.activities <= 2)


def test_all_rejected_batches_still_advance_plateau():
    from sequence_optimization.saturation import (
        _SaturationMonitor, _TrackingScorer, _SaturationReached,
    )
    wrapped = ObjectiveScorer(
        CallableScorer(lambda seqs: [0. if s == "AAAA" else 7. for s in seqs]),
        ["maximize"], stop_values=[(-2, 6)],
    )
    monitor = _SaturationMonitor(1, 1, 2, 1e-2, 0, 1e-3)
    scorer = _TrackingScorer(wrapped, None, monitor=monitor, count_repeated=True)
    scorer.score(["AAAA"])
    scorer.score(["GGGG"])
    with pytest.raises(_SaturationReached):
        scorer.score(["CCCC"])
    assert list(monitor.archive) == ["AAAA"]


@pytest.mark.parametrize("algorithm", ["simulated_annealing", "ism_greedy", "beam_search"])
def test_mutation_parents_remain_in_bounds(monkeypatch, algorithm):
    from sequence_optimization import SequenceSpace

    observed = []
    for name in ("mutate_one", "single_mutants"):
        original = getattr(SequenceSpace, name)

        def checked(self, sequences, *args, _original=original, **kwargs):
            observed.extend(sequences)
            assert all(s.count("G") <= 2 for s in sequences)
            return _original(self, sequences, *args, **kwargs)

        monkeypatch.setattr(SequenceSpace, name, checked)
    optimize(CallableScorer(lambda seqs: [s.count("G") for s in seqs]),
             4, ["GGAA", "GGGG"], ["maximize"], 2, algorithm=algorithm,
             stop_values=[2], budget=128, diversity=0)
    assert observed
