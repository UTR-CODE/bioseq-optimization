"""Optional end-to-end checks against the separate bioseq-models package."""

import numpy as np
import pytest

from sequence_optimization import (
    DifferentiableTorchSequenceScorer,
    OptimizationConfig,
    SequenceSpace,
    available_algorithms,
    run,
)


bioseq_models = pytest.importorskip("bioseq_models")

INITIAL_SEQUENCES = (
    "CCACTCGATTAACATGTTAACAACATACTCGTCCGGCCGATCAGCGGCTA",
    "GGTACCTCTAGTTTAATCCCTCCCTTATACGGTGCAACCCGCTCCCCTCA",
    "TGCATTTTTTTCACATCTATTTGCGTTTCACATCTTTTCTTACCTCTTGA",
    "CTCATTTTTCACCTCCCTTTTATCTCTCTGCTTACGTTTCGATCCCATCA",
)
BLACK_BOX_ALGORITHMS = (
    "random",
    "ism_greedy",
    "simulated_annealing",
    "genetic_algorithm",
    "adalead",
    "beam_search",
    "cross_entropy_method",
    "cbas",
)
DIFFERENTIABLE_ALGORITHMS = (
    "vanilla_activation_maximization",
    "fast_seqprop",
    "den",
)


@pytest.fixture(scope="module")
def optimus():
    return bioseq_models.load_model("optimus5prime", device="cpu")


@pytest.fixture(scope="module")
def differentiable_optimus(optimus):
    return DifferentiableTorchSequenceScorer(
        optimus.model,
        optimus.encoder,
        device=optimus.device,
        task_names=optimus.task_names,
        soft_input_transform=lambda probabilities: probabilities.transpose(1, 2),
    )


def optimization_config(algorithm):
    # One greedy ISM sweep needs 3 * 50 mutants in addition to four seeds.
    budget = 154 if algorithm == "ism_greedy" else 12
    return OptimizationConfig(
        budget=budget,
        num_candidates=4,
        mutation_rate=0.1,
        elite_fraction=0.5,
        generator_pretrain_epochs=1,
        generator_update_epochs=1,
        cbas_batch_size=4,
        den_seed_pretrain_epochs=1,
    )


def assert_single_task_result(result):
    assert len(result.sequences) == len(set(result.sequences.tolist())) == 4
    assert all(set(sequence) <= set("ACGT") for sequence in result.sequences)
    assert result.scores.shape == (4,)
    assert result.activities.shape == (4, 1)
    assert result.task_names == ("mrl",)


def test_integration_suite_covers_every_registered_algorithm():
    tested = set(BLACK_BOX_ALGORITHMS) | set(DIFFERENTIABLE_ALGORITHMS)
    assert tested == set(available_algorithms())


@pytest.mark.parametrize("algorithm", BLACK_BOX_ALGORITHMS)
def test_optimus_runs_black_box_algorithm_directly(optimus, algorithm):
    config = optimization_config(algorithm)
    result = run(
        algorithm,
        optimus,
        SequenceSpace(50),
        config,
        initial=INITIAL_SEQUENCES,
        seed=7,
    )

    assert_single_task_result(result)
    assert 4 <= result.query_count <= config.budget


def test_differentiable_wrapper_preserves_optimus_predictions(
    optimus, differentiable_optimus
):
    expected = optimus.score(INITIAL_SEQUENCES)
    observed = differentiable_optimus.score(INITIAL_SEQUENCES).scores
    np.testing.assert_allclose(observed, expected, rtol=1e-6)


@pytest.mark.parametrize("algorithm", DIFFERENTIABLE_ALGORITHMS)
def test_optimus_runs_differentiable_algorithm(differentiable_optimus, algorithm):
    config = optimization_config(algorithm)
    result = run(
        algorithm,
        differentiable_optimus,
        SequenceSpace(50),
        config,
        initial=INITIAL_SEQUENCES,
        seed=7,
    )

    assert_single_task_result(result)
    assert result.query_count == config.budget
