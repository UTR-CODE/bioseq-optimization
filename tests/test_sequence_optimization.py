import numpy as np
import pytest

from sequence_optimization import (
    BudgetExceeded, CachedScorer, CallableScorer, OptimizationConfig,
    ISM_optimize, ScoreBatch, SequenceSpace, TargetObjective,
    available_algorithms, genetic_algorithm_optimize, load_seed_sequences,
    random_seed_sequences, run,
)


def test_sequence_space_and_seed_sources(tmp_path):
    space = SequenceSpace(4, "ACGT")
    values = random_seed_sequences(space, 10, seed=4)
    assert len(values) == len(set(values.tolist())) == 10
    path = tmp_path / "seeds.txt"
    path.write_text("# comments\nACGT\nTGCA\n")
    assert load_seed_sequences(path, space).tolist() == ["ACGT", "TGCA"]


def test_cached_scorer_enforces_unique_budget():
    scorer = CachedScorer(CallableScorer(lambda values: np.asarray([value.count("G") for value in values])), budget=2)
    assert scorer.score(["AAAA", "GGGG"]).scores.tolist() == [0, 4]
    assert scorer.score(["GGGG", "AAAA"]).scores.tolist() == [4, 0]
    assert scorer.query_count == 2
    with pytest.raises(BudgetExceeded):
        scorer.score(["CCCC"])


def test_builtin_algorithms_share_common_result_contract():
    space = SequenceSpace(6, "ACGT")
    initial = np.asarray(["A" * 6, "C" * 6, "G" * 6, "T" * 6])
    config = OptimizationConfig(budget=40, num_candidates=4)
    scorer = CallableScorer(lambda values: np.asarray([value.count("G") for value in values]))
    for name in available_algorithms():
        result = run(name, scorer, space, config, initial=initial, seed=9)
        assert len(result.sequences) == len(result.scores) == len(result.uncertainties) == 4
        assert len(set(result.sequences.tolist())) == 4
        assert result.query_count <= config.budget


def test_score_batch_rejects_bad_uncertainty_shape():
    with pytest.raises(ValueError):
        ScoreBatch([1, 2], [0])


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
        "sequence", "score", "uncertainty", "activity_target", "activity_other"
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
