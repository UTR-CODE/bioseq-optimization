from tests.helpers import run_kernel

import inspect

import numpy as np
import pytest

import sequence_optimization as so
from sequence_optimization.constraints import ConstraintEvaluator
from sequence_optimization.constraints import evaluate_constraint


def test_multiple_constraints_filter_by_intersection_before_model_scoring():
    model_queries = []

    def score(sequences):
        model_queries.extend(map(str, sequences))
        return np.asarray([sequence.count("G") for sequence in sequences])

    def gc_at_most_half(sequences):
        return np.asarray(
            [
                (sequence.count("G") + sequence.count("C")) / len(sequence) <= 0.5
                for sequence in sequences
            ],
            dtype=float,
        )

    def excludes_aa(sequence):
        return float(sequence.find("AA") == -1)

    result = run_kernel(
        so.CallableScorer(score),
        length=6,
        initial=["GGGGGG", "AAAAAA", "CCCCCC", "GCGCGC"],
        objective=["maximize"],
        constraint=[gc_at_most_half, excludes_aa],
        num_candidates=4,
        diversity=0,
        algorithm='genetic_algorithm', budget=80,
        mutation_rate=0.3,
        seed=7,
    )

    assert model_queries
    assert all(gc_at_most_half([sequence])[0] == 1 for sequence in model_queries)
    assert all(excludes_aa(sequence) == 1 for sequence in model_queries)
    assert result.constraint_names == ("gc_at_most_half", "excludes_aa")
    assert result.constraint_scores.shape == (4, 2)
    assert np.all(result.constraint_scores == 1)
    assert {
        "constraint_gc_at_most_half",
        "constraint_excludes_aa",
    } <= set(result.to_frame().columns)
    metadata = result.metadata["constraint"]
    assert metadata["mode"] == "independent_intersection"
    assert metadata["model_query_count"] == len(model_queries)
    assert metadata["model_query_count"] <= metadata["screened_query_count"]
    assert metadata["counted_query_count"] == result.query_count


def test_independent_hard_constraint_pass_counts_and_intersection():
    sequences = np.asarray(["AAAA", "AAAC", "AACC", "ACCC", "CCCC"])

    def first(values):
        return np.asarray([value.count("A") >= 2 for value in values], dtype=float)

    def second(values):
        return np.asarray([value.count("C") >= 2 for value in values], dtype=float)

    evaluator = ConstraintEvaluator([first, second], np.random.default_rng(3))
    np.testing.assert_array_equal(
        evaluator.evaluate(sequences), [False, False, True, False, False]
    )
    summary = evaluator.describe()
    assert [item["passed_count"] for item in summary["constraints"]] == [3, 3]
    assert summary["intersection_passed_count"] == 1
    assert summary["intersection_pass_rate"] == pytest.approx(0.2)


def test_single_sequence_constraint_is_adapted_for_one_row_batches():
    def excludes_motif(sequence):
        return float("AAAA" not in sequence)

    np.testing.assert_array_equal(
        evaluate_constraint(excludes_motif, ["CAAAAT"]), [0.0]
    )


def test_soft_constraints_are_reproducible_and_report_probabilities():
    def soft_gc(values):
        return np.asarray(
            [
                max(0.2, 1.0 - abs(value.count("G") / len(value) - 0.5))
                for value in values
            ]
        )

    scorer = so.CallableScorer(
        lambda values: np.asarray([value.count("G") for value in values])
    )
    options = dict(
        length=5,
        initial=None,
        objective=["maximize"],
        constraint=soft_gc,
        num_candidates=4,
        diversity=0,
        budget=40,
        seed=11,
    )
    first = run_kernel(scorer, algorithm="random", **options)
    second = run_kernel(scorer, algorithm="random", **options)
    np.testing.assert_array_equal(first.sequences, second.sequences)
    np.testing.assert_allclose(first.constraint_scores, second.constraint_scores)
    expected = soft_gc(first.sequences)
    np.testing.assert_allclose(first.constraint_scores[:, 0], expected)


@pytest.mark.parametrize(
    "constraint,match",
    [
        (lambda values: np.ones((len(values), 1)), "shape"),
        (lambda values: np.full(len(values), 1.1), r"\[0, 1\]"),
        (lambda values: np.full(len(values), np.nan), "finite"),
    ],
)
def test_constraint_output_validation(constraint, match):
    with pytest.raises(ValueError, match=match):
        run_kernel(
            so.CallableScorer(lambda values: np.zeros(len(values))),
            length=3,
            initial=None,
            objective=["maximize"],
            constraint=constraint,
            num_candidates=2,
            diversity=0,
            algorithm='random', budget=8,
            seed=3,
        )


def test_constraint_requires_callable_and_enough_feasible_sequences():
    scorer = so.CallableScorer(lambda values: np.zeros(len(values)))
    with pytest.raises(TypeError, match="callable"):
        run_kernel(
            scorer,
            length=2,
            initial=None,
            objective=["maximize"],
            constraint=[lambda values: np.ones(len(values)), "invalid"],
            num_candidates=2,
            diversity=0,
            algorithm='random', budget=8,
        )
    with pytest.raises(ValueError, match="enough unique initial sequences"):
        run_kernel(
            scorer,
            length=2,
            initial=None,
            objective=["maximize"],
            constraint=lambda values: np.zeros(len(values)),
            num_candidates=2,
            diversity=0,
            algorithm='random', budget=8,
        )


def test_saturation_uses_constraint_intersection():
    def no_t(values):
        return np.asarray(["T" not in value for value in values], dtype=float)

    def no_c(values):
        return np.asarray(["C" not in value for value in values], dtype=float)

    result = so.genetic_algorithm_optimize(
        so.CallableScorer(
            lambda values: np.asarray([value.count("G") for value in values])
        ),
        length=4,
        initial=None,
        objective=["maximize"],
        constraint=[no_t, no_c],
        num_candidates=4,
        diversity=0,
        saturation=so.SaturationConfig(
            minimum_queries=8,
            plateau_patience_queries=8,
                    ),
        seed=5,
    )
    for selection in result.selections.values():
        assert all(set(sequence) <= {"A", "G"} for sequence in selection.sequences)
        assert selection.constraint_scores.shape == (4, 2)


def test_all_public_optimizers_expose_constraint_parameter():
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
        assert inspect.signature(function).parameters["constraint"].default is None
