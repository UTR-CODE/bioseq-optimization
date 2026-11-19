from tests.helpers import run_kernel

import inspect

import numpy as np
import pytest

from sequence_optimization import (
    BudgetExceeded,
    CachedScorer,
    CallableScorer,
    SequencePredictor,
    ISM_optimize,
    OptimizationConfig,
    SaturationConfig,
    ScoreBatch,
    SequenceSpace,
    TargetObjective,
    available_algorithms,
    genetic_algorithm_optimize,
    load_seed_sequences,
    pad_nucleotide_sequences,
    random_optimize,
    random_seed_sequences,
)
from sequence_optimization.api import resolve_fixed_positions
from sequence_optimization.algorithms import _hamming_distance, _run_budget_segment as run, _select_diverse


def test_sequence_space_and_seed_sources(tmp_path):
    space = SequenceSpace(4, "ACGT")
    values = random_seed_sequences(space, 10, seed=4)
    assert len(values) == len(set(values.tolist())) == 10
    path = tmp_path / "seeds.txt"
    path.write_text("# comments\nACGT\nTGCA\n")
    assert load_seed_sequences(path, space).tolist() == ["ACGT", "TGCA"]


def test_diversity_selection_recovers_from_a_blocking_top_sequence():
    ranked = [
        ("AA", (3.0, 0.0)),
        ("AC", (2.0, 0.0)),
        ("CA", (1.0, 0.0)),
    ]
    assert [item[0] for item in _select_diverse(ranked, 2, 2)] == ["AC", "CA"]


def test_hamming_distance_is_exact_for_dna_strings():
    values = ["ACGTAC", "AAAAAA", "TTTTTT", "CGTACG"]
    for left in values:
        for right in values:
            assert _hamming_distance(left, right) == sum(
                a != b for a, b in zip(left, right)
            )


def test_select_diverse_preserves_greedy_order_for_long_sequences():
    ranked = [
        ("A" * 500, (4.0, 0.0)),
        ("A" * 250 + "C" * 250, (3.0, 0.0)),
        ("C" * 500, (2.0, 0.0)),
        ("G" * 500, (1.0, 0.0)),
    ]
    selected = _select_diverse(ranked, 3, 250)
    assert [item[0] for item in selected] == [ranked[i][0] for i in (0, 1, 2)]


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
        diversity=0,
        generator_pretrain_epochs=1,
        generator_update_epochs=1,
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
    }

    def no_t(values):
        return np.asarray(["T" not in value for value in values], dtype=float)

    for name in black_box:
        result = run(
            name,
            scorer,
            space,
            config,
            initial=initial,
            constraint=no_t,
            seed=9,
        )
        assert (
            len(result.sequences)
            == len(result.scores)
            == len(result.uncertainties)
            == 4
        )
        assert len(set(result.sequences.tolist())) == 4
        assert all("T" not in sequence for sequence in result.sequences)
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

    result = run_kernel(
        SingleTaskModel(),
        SequenceSpace(4),
        algorithm='genetic_algorithm', budget=4,
        num_candidates=4,
        diversity=0,
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        objective=["maximize"],
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

    result = run_kernel(
        ScoreBatchModel(),
        SequenceSpace(4),
        algorithm='genetic_algorithm', budget=4,
        num_candidates=4,
        diversity=0,
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        objective=["maximize"],
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
    config = OptimizationConfig(budget=2, num_candidates=2, diversity=0)
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
    common = dict(
        score_threshold=0,
        num_candidates=4,
        diversity=0,
        initial=initial,
        objective=["maximize"],
        seed=3,
    )
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
            OptimizationConfig(budget=2, num_candidates=2, diversity=0),
            initial=["AA", "CC"],
            objective=TargetObjective(target="missing"),
        )


def test_random_initialization_is_unique_even_in_tiny_space():
    result = run(
        "random",
        CallableScorer(lambda values: [value == "C" for value in values]),
        SequenceSpace(1, "AC"),
        OptimizationConfig(budget=2, num_candidates=2, diversity=0),
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

    class ChannelsLastPredictor(SequencePredictor):
        def prepare_input(self, values):
            return values

    scorer = ChannelsLastPredictor(
        SumModel(),
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
            assert not torch.any(inputs.argmax(dim=-1) == 3)
            return torch.stack(
                [inputs[:, :, 2].sum(dim=1), inputs[:, :, 1].sum(dim=1)], dim=1
            )

    class ChannelsLastPredictor(SequencePredictor):
        def prepare_input(self, values):
            return values

    scorer = ChannelsLastPredictor(
        TwoTaskModel(),
        device="cpu",
        task_names=("g_activity", "c_activity"),
    )
    initial = ["AAAA", "ACCC", "AGGG", "ATTT"]
    config = OptimizationConfig(
        budget=12,
        num_candidates=4,
        diversity=0,
        fixed_region=["0:1"],
        learning_rate=0.2,
        den_seed_pretrain_epochs=2,
    )
    objective = TargetObjective(target="g_activity", mode="specificity")

    def no_t(values):
        return np.asarray(["T" not in value for value in values], dtype=float)

    for name in ("vanilla_activation_maximization", "fast_seqprop", "den", "ledidi"):
        rng_state = torch.random.get_rng_state()
        result = run(
            name,
            scorer,
            SequenceSpace(4),
            config,
            initial=initial,
            objective=objective,
            constraint=no_t,
            seed=5,
        )
        assert result.query_count == 12
        assert all(sequence[0] == "A" for sequence in result.sequences)
        assert all("T" not in sequence for sequence in result.sequences)
        assert result.activities.shape == (4, 2)
        assert result.metadata["objective"]["target_name"] == "g_activity"
        assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_saturation_returns_requested_q80_and_q90_archives():
    result = genetic_algorithm_optimize(
        CallableScorer(lambda values: [value.count("G") for value in values]),
        SequenceSpace(5),
        num_candidates=4,
        initial=["AAAAA", "CCCCC", "GGGGG", "TTTTT"],
        objective=["maximize"],
        saturation=SaturationConfig(
            fractions=(0.8, 0.9),
            minimum_queries=8,
            plateau_patience_queries=8,
            material_improvement=1e-2,
        ),
        seed=4,
    )
    assert set(result.selections) == {0.8, 0.9}
    assert result[0.8].query_count <= result[0.9].query_count
    assert result[0.9].metadata["total_convergence_queries"] == result.total_query_count


def test_score_threshold_and_gain_are_mutually_exclusive():
    scorer = CallableScorer(lambda values: [value.count("G") for value in values])
    with pytest.raises(ValueError, match="Choose one"):
        genetic_algorithm_optimize(
            scorer, 5, None, ["maximize"], 4, score_threshold=3,
            gain_fractions=(0.5, 0.9),
        )


def test_random_search_stops_when_it_reaches_the_requested_score():
    scorer = CallableScorer(lambda values: [value.count("G") for value in values])
    result = random_optimize(
        scorer, 1, ["A"], ["maximize"], 1, score_threshold=1, diversity=0, seed=4,
    )
    assert result.sequences.tolist() == ["G"]
    assert result.metadata["stop_reason"] == "target_reached"


def test_random_search_counts_repeated_proposals_with_cached_scorer():
    scorer = CachedScorer(
        CallableScorer(lambda values: [value.count("G") for value in values]),
        budget=12,
    )
    result = run(
        "random",
        scorer,
        length=1,
        config=OptimizationConfig(budget=12, num_candidates=1, diversity=0),
        initial=["A"],
        seed=4,
    )
    assert result.query_count == 12
    assert scorer.model_query_count <= 4


def test_default_diversity_applies_to_returned_sequences():
    scorer = CallableScorer(lambda values: [value.count("G") for value in values])
    result = run_kernel(
        scorer,
        length=12,
        initial=None,
        objective=["maximize"],
        num_candidates=4,
        algorithm='random', budget=200,
        seed=4,
    )
    for index, left in enumerate(result.sequences):
        for right in result.sequences[index + 1 :]:
            assert sum(a != b for a, b in zip(left, right)) >= 5
    assert result.metadata["diversity"] == 5


def test_default_diversity_applies_to_saturation_selections():
    scorer = CallableScorer(lambda values: [value.count("G") for value in values])
    saturated = genetic_algorithm_optimize(
        scorer,
        length=12,
        initial=None,
        objective=["maximize"],
        num_candidates=4,
        saturation=SaturationConfig(
            fractions=(0.8,),
            minimum_queries=4,
            plateau_patience_queries=4,
            material_improvement=100,
        ),
        seed=4,
    )
    selected = saturated[0.8]
    for index, left in enumerate(selected.sequences):
        for right in selected.sequences[index + 1 :]:
            assert sum(a != b for a, b in zip(left, right)) >= 5
    assert selected.metadata["diversity"] == 5


def test_diversity_cannot_exceed_sequence_length_for_multiple_results():
    scorer = CallableScorer(lambda values: np.zeros(len(values)))
    with pytest.raises(ValueError, match="mutable positions"):
        run_kernel(
            scorer,
            length=4,
            initial=None,
            objective=["maximize"],
            num_candidates=2,
            algorithm='random', budget=10,
        )


def test_fixed_region_resolves_requested_positive_and_negative_slices():
    positions = resolve_fixed_positions(
        ["1:3", "-3:-1", "-35:-20", slice(30, 32), -1], 50
    )
    assert positions == tuple(
        sorted(
            set(range(1, 3))
            | set(range(47, 49))
            | set(range(15, 30))
            | set(range(30, 32))
            | {49}
        )
    )


def test_fixed_region_is_preserved_by_every_black_box_optimizer():
    initial = np.asarray(["AAAACCTT", "AACCAATT", "AAGGGGTT", "AATTTGTT"])
    scorer = CallableScorer(lambda values: [value.count("G") for value in values])
    config = OptimizationConfig(
        budget=40,
        num_candidates=4,
        diversity=0,
        fixed_region=["0:2", "-2:"],
        generator_pretrain_epochs=1,
        generator_update_epochs=1,
    )
    names = (
        "basesearch",
        "random",
        "ism_greedy",
        "simulated_annealing",
        "genetic_algorithm",
        "adalead",
        "beam_search",
        "cross_entropy_method",
    )
    for name in names:
        result = run(
            name,
            scorer,
            SequenceSpace(8),
            config,
            initial=initial,
            seed=7,
        )
        assert all(sequence[:2] == "AA" for sequence in result.sequences)
        assert all(sequence[-2:] == "TT" for sequence in result.sequences)
        assert result.metadata["fixed_positions"] == [0, 1, 6, 7]


def test_fixed_region_uses_one_seeded_template_without_initial():
    result = run_kernel(
        CallableScorer(lambda values: [value.count("G") for value in values]),
        length=10,
        initial=None,
        objective=["maximize"],
        fixed_region=["0:2", "-2:"],
        num_candidates=4,
        diversity=0,
        algorithm='random', budget=20,
        seed=7,
    )
    fixed_values = {(sequence[:2], sequence[-2:]) for sequence in result.sequences}
    assert len(fixed_values) == 1
    assert result.metadata["fixed_positions"] == [0, 1, 8, 9]


def test_fixed_region_is_preserved_in_saturation_selections():
    saturated = genetic_algorithm_optimize(
        CallableScorer(lambda values: [value.count("G") for value in values]),
        length=10,
        initial=None,
        objective=["maximize"],
        fixed_region=["0:2", "-2:"],
        num_candidates=4,
        diversity=0,
        saturation=SaturationConfig(
            fractions=(0.8,),
            minimum_queries=4,
            plateau_patience_queries=4,
            material_improvement=100,
        ),
        seed=7,
    )
    selected = saturated[0.8]
    fixed_values = {(sequence[:2], sequence[-2:]) for sequence in selected.sequences}
    assert len(fixed_values) == 1
    assert selected.metadata["fixed_positions"] == [0, 1, 8, 9]


def test_fixed_region_limits_possible_diversity():
    scorer = CallableScorer(lambda values: np.zeros(len(values)))
    with pytest.raises(ValueError, match="2 mutable positions"):
        run_kernel(
            scorer,
            length=8,
            initial=None,
            objective=["maximize"],
            fixed_region=["0:6"],
            num_candidates=2,
            diversity=3,
            algorithm='random', budget=4,
        )


@pytest.mark.parametrize(
    "fixed_region,match",
    [
        ("1:3", "must be a list"),
        ([slice(1, 3, 2)], "do not support a step"),
        (["3:3"], "is empty"),
        ([10], "outside length"),
        ([[1, 2, 3]], "must contain exactly"),
        ([[1, "a"]], "interval bounds must be integers"),
        ([{1, 2}], "integers or slices"),
    ],
)
def test_invalid_fixed_region_entries(fixed_region, match):
    scorer = CallableScorer(lambda values: np.zeros(len(values)))
    with pytest.raises(ValueError, match=match):
        run_kernel(
            scorer,
            length=8,
            initial=None,
            objective=["maximize"],
            fixed_region=fixed_region,
            num_candidates=1,
            algorithm='random', budget=1,
        )


def test_fixed_region_requires_initial_sequences_to_agree():
    scorer = CallableScorer(lambda values: np.zeros(len(values)))
    with pytest.raises(ValueError, match="must agree"):
        run_kernel(
            scorer,
            length=6,
            fixed_region=["0:2"],
            initial=["AAAAAA", "CAAAAA"],
            objective=["maximize"],
            num_candidates=2,
            diversity=0,
            algorithm='random', budget=2,
        )


@pytest.mark.parametrize("diversity", [-1, 1.5, True])
def test_invalid_diversity(diversity):
    with pytest.raises(ValueError, match="diversity must be a non-negative integer"):
        OptimizationConfig(10, num_candidates=4, diversity=diversity)


def test_public_optimizers_expose_saturation_parameter():
    import sequence_optimization as so

    assert not hasattr(so, "optimize_to_saturation")
    functions = [
        so.random_optimize,
        so.genetic_algorithm_optimize,
        so.ism_optimize,
        so.simulated_annealing_optimize,
        so.adalead_optimize,
        so.beam_search_optimize,
        so.cross_entropy_optimize,
        so.basesearch_optimize,
        so.vanilla_activation_maximization_optimize,
        so.fast_seqprop_optimize,
        so.den_optimize,
        so.ledidi_optimize,
    ]
    for function in functions:
        signature = inspect.signature(function)
        required = [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
        ]
        assert required == [
            "model",
            "length",
            "initial",
            "objective",
            "num_candidates",
        ]
        assert signature.parameters["saturation"].default is None
        assert "budget" not in signature.parameters
        assert signature.parameters["score_threshold"].default is None
        assert "model_weights" not in signature.parameters
        assert signature.parameters["is_CDS_optimized"].default is False
        assert signature.parameters["diversity"].default == 5
        assert signature.parameters["fixed_region"].default is None
