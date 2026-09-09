"""Real three-model compatibility checks; small budgets, not efficacy tests."""

import numpy as np
import pytest

from sequence_optimization import (
    CombinedScorer,
    DifferentiableTorchSequenceScorer,
    OptimizationConfig,
    available_algorithms,
    run,
)

bioseq_models = pytest.importorskip("bioseq_models")
GRADIENT = {"ledidi", "vanilla_activation_maximization", "fast_seqprop", "den"}


@pytest.fixture(scope="module", params=["lentimpra", "mouse"])
def family(request):
    if request.param == "lentimpra":
        names = ["agarwal_k562", "agarwal_hepg2", "agarwal_wtc11"]
        length = 230

        def transform(values):
            return values.transpose(1, 2)[:, [0, 2, 1, 3], :]
    else:
        names = [
            "deepstarr_mouse_heart_activity",
            "deepstarr_mouse_limb_activity",
            "deepstarr_mouse_midbrain_activity",
        ]
        length = 1001

        def transform(values):
            return values.transpose(1, 2)

    models = [
        bioseq_models.load_model(name, device="cpu", batch_size=8) for name in names
    ]
    soft = [
        DifferentiableTorchSequenceScorer(
            model.model,
            model.encoder,
            device="cpu",
            batch_size=8,
            task_names=model.task_names,
            soft_input_transform=transform,
        )
        for model in models
    ]
    return models, soft, names, length


@pytest.mark.parametrize("algorithm", available_algorithms())
def test_all_algorithms_with_three_models(family, algorithm):
    models, soft, names, length = family
    scorer = CombinedScorer(soft if algorithm in GRADIENT else models, names=names)
    config = OptimizationConfig(
        4 + length * 3 if algorithm == "ism_greedy" else 12,
        num_candidates=4,
        generator_pretrain_epochs=1,
        generator_update_epochs=1,
        cbas_batch_size=4,
        den_seed_pretrain_epochs=1,
    )
    result = run(
        algorithm,
        scorer,
        length,
        config,
        objective=["maximize", "minimize", "minimize"],
        seed=7,
    )
    assert len(set(result.sequences)) == 4
    assert 4 < result.query_count <= config.budget
    expected = np.column_stack(
        [model.score(result.sequences).reshape(-1) for model in models]
    )
    np.testing.assert_allclose(result.activities, expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        result.scores, np.min(result.activities * [1, -1, -1], axis=1)
    )
    assert result.task_names == tuple(names)
    counts = list(scorer.model_query_counts.values())
    # Repeated proposals may consume budget while cached scores avoid forwards.
    assert len(set(counts)) == 1
    assert 4 <= counts[0] <= result.query_count
    if algorithm in GRADIENT:
        assert min(scorer.model_soft_query_counts.values()) > 0
