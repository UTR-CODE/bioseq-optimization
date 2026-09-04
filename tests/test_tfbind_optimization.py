import numpy as np
import pandas as pd

from BioSeqBench.optimization.tfbind8_benchmark import (
    TFBind8OptimizationConfig,
    _adalead_parent_mask,
    _aggregate_replicate_results,
    _shared_initial_codes,
    _all_single_mutants,
    optimize_adalead,
    optimize_genetic_algorithm,
    optimize_ism_greedy,
    optimize_random_screening,
    optimize_simulated_annealing,
    optimize_training_top100,
)
from BioSeqBench.optimization.tfbind8_saturation_benchmark import (
    SaturationTracker,
    TFBind8SaturationConfig,
)


def _toy_inputs():
    # A deterministic length-8 landscape whose surrogate optimum is TTTTTTTT.
    codes = np.arange(4**8, dtype=np.int64)
    scores = codes.astype(float) / codes.max()
    stds = np.full_like(scores, 0.05)
    ranked = codes.copy()
    config = TFBind8OptimizationConfig(
        train_fractions=(0.01,),
        algorithms=("random_screening",),
        num_candidates=10,
        query_budget=500,
        top_visible_initializations=5,
        random_initializations=5,
    )
    return scores, stds, ranked, int(round(len(codes) * 0.01)), config


def test_single_mutant_neighborhood_is_complete_and_unique():
    mutants = _all_single_mutants(0, 8)
    assert len(mutants) == 24
    assert len(np.unique(mutants)) == 24
    encoded = np.zeros((1, 8), dtype=np.uint8)
    mutant_encoded = np.asarray(
        [[int(char) for char in np.base_repr(code, base=4).zfill(8)] for code in mutants]
    )
    assert np.all((mutant_encoded != encoded).sum(axis=1) == 1)


def test_adalead_threshold_handles_negative_fitness():
    values = np.asarray([-2.0, -1.0, -0.9])
    mask = _adalead_parent_mask(values, threshold=0.05)
    assert mask.tolist() == [False, False, True]


def test_training_top100_returns_highest_visible_sequences_without_queries():
    scores, stds, ranked, visible_count, config = _toy_inputs()
    result = optimize_training_top100(scores, stds, ranked, visible_count, config)
    expected = ranked[visible_count - config.num_candidates : visible_count][::-1]
    assert result.query_count == 0
    assert np.array_equal(result.codes, expected)
    assert np.all(np.isin(result.codes, ranked[:visible_count]))


def test_shared_initialization_uses_train_top_and_unconditioned_full_space_random():
    landscape_size = 4**8
    ranked = np.arange(landscape_size, dtype=np.int64)
    visible_count = int(round(landscape_size * 0.95))
    config = TFBind8OptimizationConfig(
        train_fractions=(0.95,),
        algorithms=("ism_greedy",),
        num_candidates=100,
        query_budget=500,
        top_visible_initializations=50,
        random_initializations=50,
    )
    codes, sources = _shared_initial_codes(
        ranked,
        visible_count,
        landscape_size,
        config,
        np.random.default_rng(1234),
    )
    expected_top = ranked[visible_count - 50 : visible_count][::-1]
    assert np.array_equal(codes[:50], expected_top)
    assert sources.tolist() == ["train_top"] * 50 + ["full_space_random"] * 50
    assert len(np.unique(codes)) == 100
    # With 95% of the full landscape visible, a uniform full-space draw should
    # include training sequences. This distinguishes it from the old policy.
    assert np.isin(codes[50:], ranked[:visible_count]).any()


def test_black_box_optimizers_obey_budget_and_candidate_contract():
    scores, stds, ranked, visible_count, config = _toy_inputs()
    functions = (
        optimize_random_screening,
        optimize_ism_greedy,
        optimize_simulated_annealing,
        optimize_genetic_algorithm,
        optimize_adalead,
    )
    visible = set(map(int, ranked[:visible_count]))
    for offset, function in enumerate(functions):
        result = function(scores, stds, ranked, visible_count, 8, config, 100 + offset)
        assert result.query_count == config.query_budget
        assert len(result.codes) == config.num_candidates
        assert len(np.unique(result.codes)) == config.num_candidates
        assert not visible.intersection(map(int, result.codes))


def test_replicate_aggregation_computes_paired_random_gain():
    rows = []
    for replicate, random_score, ism_score in ((0, 0.1, 0.3), (1, 0.2, 0.5)):
        for algorithm, score in (("random_screening", random_score), ("ism_greedy", ism_score)):
            rows.append(
                {
                    "dataset": "tfbind8",
                    "weak_model_fraction": 0.5,
                    "algorithm": algorithm,
                    "replicate": replicate,
                    "mean_true_score": score,
                    "median_true_score": score,
                    "top_10_mean_true_score": score,
                    "best_true_score": score,
                    "top_5pct_count": 0,
                    "top_1pct_count": 0,
                    "top_0_1pct_count": 0,
                    "mean_pairwise_hamming": 4.0,
                    "mean_minimum_training_hamming": 1.0,
                    "mean_absolute_prediction_error": 0.1,
                }
            )
    aggregate = _aggregate_replicate_results(pd.DataFrame(rows))
    ism = aggregate[aggregate["algorithm"] == "ism_greedy"].iloc[0]
    assert ism["replicates"] == 2
    assert np.isclose(ism["mean_true_score_mean"], 0.4)
    assert np.isclose(ism["mean_activity_gain_over_random_mean"], 0.25)


def test_saturation_tracker_returns_first_archive_at_99_percent_gain():
    config = TFBind8SaturationConfig(
        train_fractions=(0.01,), maximum_queries=1_000, minimum_queries=900,
        plateau_patience_queries=1_000, snapshot_interval_queries=1,
    )
    tracker = SaturationTracker(set(), config)
    from BioSeqBench.optimization.tfbind8_benchmark import CandidateArchive

    archive = CandidateArchive()
    codes = np.arange(100, dtype=np.int64)
    archive.add(codes, np.zeros(100), np.zeros(100), step=0)
    tracker.observe(archive, 100)
    archive.add(codes, np.ones(100), np.zeros(100), step=1)
    tracker.observe(archive, 200)
    archive.add(np.arange(100, 200), np.zeros(100), np.zeros(100), step=2)
    tracker.observe(archive, 300)
    selected = tracker.finalize("test")
    assert selected.q99_query_count == 200
    assert selected.q99_top100_mean == 1.0
    assert selected.saturation_top100_mean == 1.0
