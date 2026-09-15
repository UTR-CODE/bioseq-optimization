import numpy as np
import pytest
import torch

import sequence_optimization as so


def test_onehot_values_layout_and_rna():
    expected = np.vstack([np.eye(4), np.zeros((1, 4))]).astype(np.float32)
    np.testing.assert_array_equal(so.onehot("acgun")[0], expected)
    np.testing.assert_array_equal(
        so.onehot([b"ACGTN"], channels_first=True)[0], expected.T
    )
    np.testing.assert_array_equal(
        so.onehot(np.array(["ACGTN"], dtype=object))[0], expected
    )
    assert so.onehot(["AC"], dtype=np.uint8).dtype == np.uint8
    np.testing.assert_array_equal(so.onehot("AGCT", alphabet="AGCT")[0], np.eye(4))
    assert so.onehot([]).shape == (0, 0, 4)
    assert so.onehot("").shape == (1, 0, 4)


@pytest.mark.parametrize("values", [["A", "AC"], "AX", "在", [["AC"]], [123]])
def test_onehot_rejects_invalid_input(values):
    with pytest.raises(ValueError):
        so.onehot(values)


def test_default_torch_onehot_preserves_predictions_and_gradients():
    class Model(torch.nn.Module):
        def forward(self, values):
            return values[:, 2].sum(dim=1)

    scorer = so.SequencePredictor(Model())
    np.testing.assert_array_equal(scorer.score(["AGGT", "AAAA"]).scores, [2, 0])
    inputs = torch.from_numpy(so.onehot(["AGGT", "AAAA"])).requires_grad_()
    scores, _ = scorer.score_soft(inputs)
    scores.sum().backward()
    assert inputs.grad[:, :, 2].eq(1).all()


@pytest.mark.parametrize("algorithm", so.available_algorithms())
def test_every_convenience_function_accepts_length(algorithm):
    class Model(torch.nn.Module):
        def forward(self, values):
            return values[:, 2].sum(dim=1)

    scorer = so.SequencePredictor(Model())
    function_name = {
        "ism_greedy": "ism_optimize",
        "cross_entropy_method": "cross_entropy_optimize",
    }.get(algorithm, algorithm + "_optimize")
    result = getattr(so, function_name)(
        scorer,
        length=8,
        initial=None,
        objective=["maximize"],
        num_candidates=4,
        budget=28,
        generator_pretrain_epochs=1,
        generator_update_epochs=1,
        den_seed_pretrain_epochs=1,
        cbas_batch_size=4,
    )
    assert all(len(sequence) == 8 for sequence in result.sequences)
    assert result.query_count <= 28


def test_length_seed_and_saturation_interfaces(tmp_path):
    sequences = so.random_seed_sequences(length=4, count=4, alphabet="AC")
    path = tmp_path / "seeds.txt"
    path.write_text("\n".join(sequences))
    np.testing.assert_array_equal(
        so.load_seed_sequences(path, length=4, alphabet="AC"), sequences
    )
    scorer = so.CallableScorer(lambda values: np.array([s.count("A") for s in values]))
    many = so.run_many(
        ["random", "basesearch"],
        scorer,
        length=4,
        config=so.OptimizationConfig(8, num_candidates=4, diversity=0),
        alphabet="AC",
    )
    assert all(set("".join(result.sequences)) <= set("AC") for result in many.values())
    result = so.genetic_algorithm_optimize(
        scorer,
        length=4,
        initial=None,
        objective=["maximize"],
        alphabet="AC",
        num_candidates=4,
        diversity=0,
        saturation=so.SaturationConfig(
            minimum_queries=4, plateau_patience_queries=4, maximum_queries=12
        ),
    )
    assert result.selections


@pytest.mark.parametrize("length", [0, -1, 1.5, True])
def test_invalid_length(length):
    with pytest.raises(ValueError):
        so.random_optimize(
            so.CallableScorer(lambda values: np.ones(len(values))),
            length=length,
            initial=None,
            objective=["maximize"],
            num_candidates=1,
            budget=1,
        )
