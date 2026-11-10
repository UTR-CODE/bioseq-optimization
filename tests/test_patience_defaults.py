"""Population-scaled plateau defaults must allow delayed hard decoding."""

import numpy as np
import pytest
import torch

from sequence_optimization import (
    CallableScorer,
    SaturationConfig,
    SequencePredictor,
    generate_designs,
)
from sequence_optimization.algorithms import optimize


@pytest.mark.parametrize(
    "population,candidates", [(20, 20), (100, 100), (100, 20), (200, 100)]
)
def test_patience_scales_with_search_population(population, candidates):
    assert SaturationConfig().resolve(candidates, population) == (
        10 * population,
        40 * population,
        None,
    )


def test_explicit_patience_is_not_rescaled():
    config = SaturationConfig(
        minimum_queries=20, plateau_patience_queries=7, maximum_queries=200
    )
    assert config.resolve(20, 100) == (20, 7, 200)


def test_dynamic_material_threshold_has_one_percent_floor():
    config = SaturationConfig()
    assert config.material_improvement == 1e-2
    assert config.relative_material_improvement == 1e-3
    with pytest.raises(ValueError, match="at least 1e-2"):
        SaturationConfig(material_improvement=0.009).resolve(20)
    with pytest.raises(ValueError, match="relative_material_improvement"):
        SaturationConfig(relative_material_improvement=-1e-3).resolve(20)


@pytest.mark.parametrize("field", ["minimum_queries", "plateau_patience_queries"])
def test_zero_query_setting_is_not_silently_defaulted(field):
    with pytest.raises(ValueError, match="query limits"):
        SaturationConfig(**{field: 0}).resolve(20)


def test_suite_default_does_not_stop_before_vanilla_argmax_changes():
    class CountG(torch.nn.Module):
        def forward(self, x):
            return x[:, 2, :].sum(dim=1, keepdim=True)

    model = SequencePredictor(CountG())
    initial = np.array(["AAAAAAAA", "CCCCCCCC", "TTTTTTTT", "ACACACAC"])
    output = generate_designs(
        model,
        8,
        initial,
        ["maximize"],
        algorithms=["vanilla_activation_maximization"],
        num_candidates=1,
        diversity=0,
        seed=42,
    )
    assert output.score.max() > 0
    assert output.sequence.iloc[0] not in set(initial)


def test_suite_cds_default_uses_per_reference_population(monkeypatch):
    import sequence_optimization.suite as suite

    captured = []

    def fake_optimize(*args, **kwargs):
        captured.append(kwargs["saturation"])
        return []

    monkeypatch.setattr(suite, "optimize", fake_optimize)
    generate_designs(
        object(),
        12,
        ["ATGGCTGCTTAA"],
        ["maximize"],
        parallel=False,
        algorithms=["genetic_algorithm"],
        num_candidates=3000,
        is_CDS_optimized=True,
    )
    assert captured[0].maximum_queries is None
    assert captured[0].selection == "knee_fraction"


@pytest.mark.parametrize(
    "algorithm,expected",
    [
        ("beam_search", 450),
        ("ism_greedy", 1800),
        ("cross_entropy_method", 768),
        ("basesearch", 768),
        ("genetic_algorithm", 160),
        ("simulated_annealing", 800),
    ],
)
def test_default_patience_covers_three_proposal_batches(algorithm, expected):
    scorer = CallableScorer(lambda values: np.zeros(len(values)))
    result = optimize(
        scorer,
        50,
        [c * 50 for c in "ACGT"],
        ["maximize"],
        4,
        algorithm=algorithm,
        diversity=0,
        saturation=SaturationConfig(maximum_queries=40),
        generator_pretrain_epochs=1,
        generator_update_epochs=1,
    )
    assert result[0.99].metadata["plateau_patience_queries"] == expected


@pytest.mark.parametrize('algorithm', ['beam_search', 'simulated_annealing'])
def test_explicit_window_bypasses_neighborhood_floor(algorithm):
    result = optimize(
        CallableScorer(lambda values: np.zeros(len(values))),
        50,
        [c * 50 for c in "ACGT"],
        ["maximize"],
        4,
        algorithm=algorithm,
        diversity=0,
        saturation=SaturationConfig(
            minimum_queries=4, plateau_patience_queries=7, maximum_queries=40
        ),
    )
    assert result[0.99].metadata["plateau_patience_queries"] == 7
