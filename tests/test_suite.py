import numpy as np
import pytest
import torch

from sequence_optimization import (
    CallableScorer,
    CombinedScorer,
    SaturationConfig,
    SequencePredictor,
    cross_entropy_optimize,
    generate_designs,
    random_seed_sequences,
)


def _count_g(values):
    return np.asarray([value.count("G") for value in values])


def test_generate_designs_returns_frame_with_one_row_per_candidate():
    initial = np.asarray(
        [
            "AAAAAA",
            "CCCCCC",
            "GGGGGG",
            "TTTTTT",
            "ACGTAC",
            "CGTACG",
            "GTACGT",
            "TACGTA",
        ]
    )
    frame = generate_designs(
        CallableScorer(_count_g),
        algorithms=["basesearch", "genetic_algorithm"],
        length=6,
        initial=initial,
        objective=["maximize"],
        num_candidates=3,
        budget=24,
        diversity=0,
        seed=7,
    )

    assert list(frame.columns[:3]) == ["sequence", "algorithm", "score"]
    assert len(frame) == 6
    assert frame.groupby("algorithm").size().to_dict() == {
        "basesearch": 3,
        "genetic_algorithm": 3,
    }
    assert frame["sequence"].str.len().eq(6).all()


def test_generate_designs_requires_gradient_model_before_running():
    model = CallableScorer(lambda values: np.zeros(len(values)))
    with pytest.raises(TypeError, match="differentiable_model is required"):
        generate_designs(
            model,
            algorithms=["random", "fast_seqprop"],
            length=4,
            initial=["AAAA", "CCCC"],
            objective=["maximize"],
            num_candidates=1,
            budget=4,
            diversity=0,
        )


def test_combined_scorer_only_advertises_soft_scoring_when_all_children_support_it():
    discrete = CallableScorer(_count_g)
    differentiable = SequencePredictor(
        torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(16, 1),
        )
    )
    assert not CombinedScorer([differentiable, discrete]).supports_soft_scoring
    assert CombinedScorer([differentiable, differentiable]).supports_soft_scoring


def test_generate_designs_unwraps_saturation_selection():
    frame = generate_designs(
        CallableScorer(_count_g),
        algorithms=["genetic_algorithm"],
        length=4,
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        objective=["maximize"],
        num_candidates=2,
        diversity=0,
        saturation=SaturationConfig(
            fractions=(0.9,),
            minimum_queries=2,
            plateau_patience_queries=1,
            maximum_queries=8,
        ),
        seed=4,
    )
    assert len(frame) == 2
    assert frame["algorithm"].eq("genetic_algorithm").all()


def test_generate_designs_shares_model_predictions_between_runs():
    queried = []

    def score(values):
        queried.extend(values)
        return _count_g(values)

    frame = generate_designs(
        CallableScorer(score),
        algorithms=["random", "genetic_algorithm"],
        length=4,
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        objective=["maximize"],
        num_candidates=2,
        budget=4,
        diversity=0,
    )
    # The shared cache means the second algorithm re-uses the first one's scores.
    assert len(queried) == len(set(queried))
    assert len(frame) == 4


def test_generate_designs_returns_requested_candidate_count():
    initial = random_seed_sequences(length=8, count=1000, seed=7)
    frame = generate_designs(
        CallableScorer(_count_g),
        algorithms=["random"],
        length=8,
        initial=initial,
        objective=["maximize"],
        num_candidates=100,
        budget=1000,
        diversity=0,
    )
    assert len(frame) == 100


def test_generate_designs_rejects_initial_smaller_than_candidate_count():
    with pytest.raises(ValueError, match="at least 10 sequences"):
        generate_designs(
            CallableScorer(_count_g),
            algorithms=["random"],
            length=4,
            initial=["AAAA", "CCCC"],
            objective=["maximize"],
            num_candidates=10,
            budget=8,
            diversity=0,
        )


def test_generate_designs_adds_activity_columns_for_multi_task_models():
    class TwoTaskModel:
        task_names = ("dev", "hk")

        def score(self, sequences):
            return np.asarray(
                [[sequence.count("G"), sequence.count("C")] for sequence in sequences],
                dtype=float,
            )

    frame = generate_designs(
        TwoTaskModel(),
        algorithms=["random"],
        length=4,
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        objective=["minimize", "maximize"],
        num_candidates=2,
        budget=4,
        diversity=0,
    )
    assert "activity_dev" in frame.columns
    assert "activity_hk" in frame.columns


def test_generate_designs_applies_objective_weights():
    class TwoTaskModel:
        task_names = ("a", "b")

        def score(self, sequences):
            return np.asarray(
                [[sequence.count("G"), sequence.count("C")] for sequence in sequences],
                dtype=float,
            )

    kwargs = dict(
        algorithms=["random"],
        length=4,
        initial=["AAAA", "CCCC", "GGGG", "TTTT", "ACGT", "CGTA"],
        objective=["maximize", "maximize"],
        num_candidates=2,
        budget=6,
        diversity=0,
        seed=3,
    )
    equal = generate_designs(TwoTaskModel(), **kwargs)
    weighted = generate_designs(TwoTaskModel(), weights=[3.0, 1.0], **kwargs)

    # Weighting task "a" 3x favors G-rich sequences over C-rich ones.
    assert weighted["score"].max() > equal["score"].max()


def test_generate_designs_rejects_weights_without_directions_objective():
    with pytest.raises(TypeError, match="weights require"):
        generate_designs(
            CallableScorer(_count_g),
            algorithms=["random"],
            length=4,
            initial=["AAAA", "CCCC"],
            objective=None,
            num_candidates=1,
            budget=4,
            diversity=0,
            weights=[1.0],
        )


def test_generate_designs_keeps_multiple_fixed_regions_unchanged():
    raw = random_seed_sequences(length=12, count=8, seed=5)
    initial = ["AC" + str(value)[2:-2] + "TT" for value in raw]

    frame = generate_designs(
        CallableScorer(_count_g),
        algorithms=["genetic_algorithm"],
        length=12,
        initial=initial,
        objective=["maximize"],
        num_candidates=3,
        budget=24,
        diversity=0,
        fixed_region=[[0, 2], [-2, None]],
        seed=11,
    )
    assert frame["sequence"].str[:2].eq("AC").all()
    assert frame["sequence"].str[-2:].eq("TT").all()


def test_generate_designs_honours_explicit_budget_over_saturation_default():
    queried = []

    def score(values):
        queried.extend(values)
        return _count_g(values)

    generate_designs(
        CallableScorer(score),
        algorithms=["genetic_algorithm"],
        length=6,
        initial=random_seed_sequences(length=6, count=8, seed=2),
        objective=["maximize"],
        num_candidates=2,
        budget=20,
        diversity=0,
        seed=2,
    )
    # The explicit budget caps queries well below the saturation default.
    assert len(queried) <= 20


def test_generate_designs_rejects_unknown_algorithm():
    with pytest.raises(KeyError, match="unknown algorithms"):
        generate_designs(
            CallableScorer(_count_g),
            algorithms=["not_an_algorithm"],
            length=4,
            initial=["AAAA", "CCCC"],
            objective=["maximize"],
            num_candidates=1,
            budget=4,
            diversity=0,
        )


def test_cross_entropy_uses_initial_population_as_proposal_batch():
    class BatchModel:
        task_names = ("activity",)

        def __init__(self):
            self.batch_sizes = []

        def score(self, sequences):
            self.batch_sizes.append(len(sequences))
            return np.asarray([[sequence.count("G")] for sequence in sequences])

    model = BatchModel()
    initial = ["AAAA", "CCCC", "GGGG", "TTTT", "ACGT", "CGTA", "GTAC", "TACG"]
    result = cross_entropy_optimize(
        model,
        length=4,
        initial=initial,
        objective=["maximize"],
        num_candidates=2,
        budget=16,
        diversity=0,
        seed=7,
    )
    assert model.batch_sizes == [8, 8]
    assert result.metadata["batch_size"] == 8
