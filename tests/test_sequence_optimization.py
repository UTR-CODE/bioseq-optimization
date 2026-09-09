import numpy as np
import pytest

from sequence_optimization import (
    BudgetExceeded,
    CachedScorer,
    CallableScorer,
    DifferentiableTorchSequenceScorer,
    ISM_optimize,
    OptimizationConfig,
    SaturationConfig,
    ScoreBatch,
    SequenceSpace,
    TargetObjective,
    available_algorithms,
    genetic_algorithm_optimize,
    load_seed_sequences,
    optimize_to_saturation,
    pad_nucleotide_sequences,
    random_seed_sequences,
    run,
)


def test_sequence_space_and_seed_sources(tmp_path):
    space = SequenceSpace(4, "ACGT")
    values = random_seed_sequences(space, 10, seed=4)
    assert len(values) == len(set(values.tolist())) == 10
    path = tmp_path / "seeds.txt"
    path.write_text("# comments\nACGT\nTGCA\n")
    assert load_seed_sequences(path, space).tolist() == ["ACGT", "TGCA"]


def test_cached_scorer_enforces_unique_budget():
    scorer = CachedScorer(
        CallableScorer(
            lambda values: np.asarray([value.count("G") for value in values])
        ),
        budget=2,
    )
    assert scorer.score(["AAAA", "GGGG"]).scores.tolist() == [0, 4]
    assert scorer.score(["GGGG", "AAAA"]).scores.tolist() == [4, 0]
    assert scorer.query_count == 2
    with pytest.raises(BudgetExceeded):
        scorer.score(["CCCC"])


def test_builtin_algorithms_share_common_result_contract():
    space = SequenceSpace(6, "ACGT")
    initial = np.asarray(["A" * 6, "C" * 6, "G" * 6, "T" * 6])
    config = OptimizationConfig(
        budget=40,
        num_candidates=4,
        generator_pretrain_epochs=1,
        generator_update_epochs=1,
        cbas_batch_size=8,
    )
    scorer = CallableScorer(
        lambda values: np.asarray([value.count("G") for value in values])
    )
    black_box = {
        "basesearch",
        "random",
        "ism_greedy",
        "simulated_annealing",
        "genetic_algorithm",
        "adalead",
        "beam_search",
        "cross_entropy_method",
        "cbas",
    }
    for name in black_box:
        result = run(name, scorer, space, config, initial=initial, seed=9)
        assert (
            len(result.sequences)
            == len(result.scores)
            == len(result.uncertainties)
            == 4
        )
        assert len(set(result.sequences.tolist())) == 4
        assert result.query_count <= config.budget
    assert black_box <= set(available_algorithms())


def test_score_batch_rejects_bad_uncertainty_shape():
    with pytest.raises(ValueError):
        ScoreBatch([1, 2], [0])


@pytest.mark.parametrize("column_output", [False, True])
def test_plain_single_task_model_can_return_numpy_array(column_output):
    class SingleTaskModel:
        task_names = ("activity",)

        def score(self, values):
            activities = np.asarray([value.count("G") for value in values])
            return activities[:, None] if column_output else activities

    result = genetic_algorithm_optimize(
        SingleTaskModel(),
        SequenceSpace(4),
        budget=4,
        num_candidates=4,
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        seed=3,
    )

    assert result.sequences[0] == "GGGG"
    assert result.scores[0] == 4
    assert result.activities.shape == (4, 1)
    assert result.task_names == ("activity",)
    assert result.query_count == 4


def test_plain_model_score_batch_output_remains_supported():
    class ScoreBatchModel:
        task_names = ("activity",)

        def score(self, values):
            return ScoreBatch([value.count("G") for value in values])

    result = genetic_algorithm_optimize(
        ScoreBatchModel(),
        SequenceSpace(4),
        budget=4,
        num_candidates=4,
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        seed=3,
    )

    assert result.sequences[0] == "GGGG"
    assert result.scores[0] == 4
    assert result.activities.shape == (4, 1)


def test_cached_scorer_preserves_multitask_outputs():
    scorer = CachedScorer(
        CallableScorer(
            lambda values: np.asarray(
                [[value.count("G"), value.count("C")] for value in values]
            ),
            task_names=("g_activity", "c_activity"),
        ),
        budget=2,
    )
    first = scorer.score(["GGGG", "CCCC"])
    repeated = scorer.score(["CCCC", "GGGG"])
    assert first.scores.shape == (2, 2)
    assert repeated.scores.tolist() == [[0, 4], [4, 0]]
    assert scorer.query_count == 2


def test_target_only_and_crested_style_specificity_rank_differently():
    space = SequenceSpace(4)
    initial = np.asarray(["AAAA", "CCCC"])

    def activities(values):
        lookup = {"AAAA": (10.0, 9.0), "CCCC": (8.0, 0.0)}
        return np.asarray([lookup[str(value)] for value in values])

    scorer = CallableScorer(activities, task_names=("target", "other"))
    config = OptimizationConfig(budget=2, num_candidates=2)
    target_only = run(
        "random",
        scorer,
        space,
        config,
        initial=initial,
        objective=TargetObjective(target="target", mode="maximize"),
    )
    specificity = run(
        "random",
        scorer,
        space,
        config,
        initial=initial,
        objective=TargetObjective(target="target", mode="specificity"),
    )

    assert target_only.sequences[0] == "AAAA"
    assert specificity.sequences[0] == "CCCC"
    assert specificity.scores.tolist() == [8.0, 5.5]
    assert specificity.activities.shape == (2, 2)
    assert specificity.task_names == ("target", "other")
    assert specificity.metadata["objective"]["coefficients"] == [1.0, -0.5]
    assert list(specificity.to_frame().columns) == [
        "sequence",
        "score",
        "uncertainty",
        "activity_target",
        "activity_other",
    ]


def test_named_optimizer_functions_and_rna_space():
    space = SequenceSpace(4, "ACGU")
    initial = np.asarray(["AAAA", "CCCC", "GGGG", "UUUU"])
    scorer = CallableScorer(lambda values: [value.count("U") for value in values])
    common = dict(budget=4, num_candidates=4, initial=initial, seed=3)
    ism = ISM_optimize(scorer, space, **common)
    genetic = genetic_algorithm_optimize(scorer, space, **common)
    assert ism.algorithm == "ism_greedy"
    assert genetic.algorithm == "genetic_algorithm"
    assert ism.activities.shape == (4, 1)
    assert all("U" not in sequence for sequence in ism.sequences)


def test_named_target_requires_scorer_task_names():
    scorer = CallableScorer(lambda values: np.ones((len(values), 2)))
    with pytest.raises(ValueError, match="requires task_names"):
        run(
            "random",
            scorer,
            SequenceSpace(2),
            OptimizationConfig(budget=2, num_candidates=2),
            initial=["AA", "CC"],
            objective=TargetObjective(target="missing"),
        )


def test_random_initialization_is_unique_even_in_tiny_space():
    result = run(
        "random",
        CallableScorer(lambda values: [value == "C" for value in values]),
        SequenceSpace(1, "AC"),
        OptimizationConfig(budget=2, num_candidates=2),
        seed=2,
    )
    assert set(result.sequences) == {"A", "C"}


def test_sequence_space_normalizes_rna_u_to_dna_t():
    space = SequenceSpace(4, "ACGU")
    assert space.alphabet == "ACGT"
    assert space.validate(["acgu", "UUUU"]).tolist() == ["ACGT", "TTTT"]
    observed = []
    scorer = CallableScorer(lambda values: observed.extend(values.tolist()) or [1, 1])
    scorer.score(["acgu", "UUUU"])
    assert observed == ["ACGT", "TTTT"]


def test_variable_sequences_can_be_left_or_right_padded_for_a_model():
    assert pad_nucleotide_sequences(["ACGU", "AC"], 6, side="right").tolist() == [
        "ACGTNN",
        "ACNNNN",
    ]
    assert pad_nucleotide_sequences(["ACGU", "AC"], 6, side="left").tolist() == [
        "NNACGT",
        "NNNNAC",
    ]

    observed = []
    scorer = CallableScorer(
        lambda values: observed.extend(values.tolist()) or [1.0, 2.0],
        pad_to_length=6,
        padding_side="left",
    )
    scorer.score(["ACGU", "AC"])
    assert observed == ["NNACGT", "NNNNAC"]


def test_soft_sequences_use_zero_vector_n_padding():
    import torch

    class SumModel(torch.nn.Module):
        def forward(self, inputs):
            assert inputs.shape == (1, 6, 4)
            return inputs.sum(dim=(1, 2))

    scorer = DifferentiableTorchSequenceScorer(
        SumModel(),
        lambda values: values,
        pad_to_length=6,
        padding_side="left",
    )
    probabilities = torch.full((1, 4, 4), 0.25, requires_grad=True)
    activities, _ = scorer.score_soft(probabilities)
    activities.sum().backward()
    assert activities.item() == pytest.approx(4.0)
    assert probabilities.grad is not None


def test_differentiable_algorithms_share_the_result_contract():
    import torch

    class TwoTaskModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.stack(
                [inputs[:, :, 2].sum(dim=1), inputs[:, :, 1].sum(dim=1)], dim=1
            )

    def encode(values):
        lookup = {base: index for index, base in enumerate("ACGT")}
        encoded = torch.tensor([[lookup[base] for base in value] for value in values])
        return torch.nn.functional.one_hot(encoded, num_classes=4).float()

    scorer = DifferentiableTorchSequenceScorer(
        TwoTaskModel(),
        encode,
        device="cpu",
        task_names=("g_activity", "c_activity"),
    )
    initial = ["AAAA", "CCCC", "GGGG", "TTTT"]
    config = OptimizationConfig(
        budget=12,
        num_candidates=4,
        learning_rate=0.2,
        den_seed_pretrain_epochs=2,
    )
    objective = TargetObjective(target="g_activity", mode="specificity")
    for name in ("vanilla_activation_maximization", "fast_seqprop", "den"):
        result = run(
            name,
            scorer,
            SequenceSpace(4),
            config,
            initial=initial,
            objective=objective,
            seed=5,
        )
        assert result.query_count == 12
        assert result.activities.shape == (4, 2)
        assert result.metadata["objective"]["target_name"] == "g_activity"


def test_saturation_returns_requested_q80_and_q90_archives():
    result = optimize_to_saturation(
        CallableScorer(lambda values: [value.count("G") for value in values]),
        SequenceSpace(5),
        algorithm="genetic_algorithm",
        num_candidates=4,
        initial=["AAAAA", "CCCCC", "GGGGG", "TTTTT"],
        saturation=SaturationConfig(
            fractions=(0.8, 0.9),
            minimum_queries=8,
            plateau_patience_queries=8,
            maximum_queries=40,
            material_improvement=1e-6,
        ),
        seed=4,
    )
    assert set(result.selections) == {0.8, 0.9}
    assert result[0.8].query_count <= result[0.9].query_count
    assert result[0.9].metadata["total_convergence_queries"] == result.total_query_count
