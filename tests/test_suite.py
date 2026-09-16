import numpy as np

from sequence_optimization import (
    CallableScorer,
    cross_entropy_optimize,
    optimize_algorithms,
    random_seed_sequences,
)


def test_optimize_algorithms_uses_large_shared_initial_population(tmp_path):
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
    suite = optimize_algorithms(
        CallableScorer(lambda values: [value.count("G") for value in values]),
        algorithms=["basesearch", "genetic_algorithm"],
        length=6,
        initial=initial,
        objective=["maximize"],
        num_candidates=3,
        budget=24,
        diversity=0,
        seed=7,
    )

    assert suite.initial_count == 8
    assert list(suite.results) == ["basesearch", "genetic_algorithm"]
    assert all(len(result.sequences) == 3 for result in suite.results.values())
    assert len(suite.to_frame()) == 6
    assert suite.summary_frame()["candidate_count"].tolist() == [3, 3]
    paths = suite.save_csv(tmp_path / "results")
    assert all(path.is_file() for path in paths.values())


def test_optimize_algorithms_requires_gradient_model_before_running():
    model = CallableScorer(lambda values: np.zeros(len(values)))
    initial = ["AAAA", "CCCC"]
    try:
        optimize_algorithms(
            model,
            algorithms=["random", "fast_seqprop"],
            length=4,
            initial=initial,
            objective=["maximize"],
            num_candidates=1,
            budget=4,
            diversity=0,
        )
    except TypeError as error:
        assert "differentiable_model is required" in str(error)
    else:
        raise AssertionError("missing differentiable_model was accepted")


def test_optimize_algorithms_shares_model_predictions_between_runs():
    queried = []

    def score(values):
        queried.extend(values)
        return np.asarray([value.count("G") for value in values])

    initial = ["AAAA", "CCCC", "GGGG", "TTTT"]
    suite = optimize_algorithms(
        CallableScorer(score),
        algorithms=["random", "genetic_algorithm"],
        length=4,
        initial=initial,
        objective=["maximize"],
        num_candidates=2,
        budget=4,
        diversity=0,
    )
    assert len(queried) == 4
    assert suite.model_query_counts == {"random": 4, "genetic_algorithm": 0}
    assert suite.max_model_batch_sizes == {"random": 4, "genetic_algorithm": 0}
    assert suite.soft_query_counts == {"random": 0, "genetic_algorithm": 0}
    assert suite.max_soft_batch_sizes == {"random": 0, "genetic_algorithm": 0}


def test_optimize_algorithms_can_disable_cross_run_cache():
    queried = []

    def score(values):
        queried.extend(values)
        return np.asarray([value.count("G") for value in values])

    suite = optimize_algorithms(
        CallableScorer(score),
        algorithms=["random", "genetic_algorithm"],
        length=4,
        initial=["AAAA", "CCCC", "GGGG", "TTTT"],
        objective=["maximize"],
        num_candidates=2,
        budget=4,
        diversity=0,
        share_model_cache=False,
    )
    assert len(queried) == 8
    assert suite.model_query_counts == {"random": 4, "genetic_algorithm": 4}
    assert suite.max_model_batch_sizes == {"random": 4, "genetic_algorithm": 4}


def test_suite_returns_100_candidates_from_1000_initial_sequences():
    initial = random_seed_sequences(length=8, count=1000, seed=7)
    suite = optimize_algorithms(
        CallableScorer(
            lambda values: np.asarray([value.count("G") for value in values])
        ),
        algorithms=["random"],
        length=8,
        initial=initial,
        objective=["maximize"],
        num_candidates=100,
        budget=1000,
        diversity=0,
    )
    assert suite.initial_count == 1000
    assert len(suite.results["random"].sequences) == 100


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
