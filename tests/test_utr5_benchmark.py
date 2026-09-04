import numpy as np
import pytest

import BioSeqBench.optimization.utr5_benchmark as benchmark
import BioSeqBench.optimization.utr5_saturation_benchmark as saturation
from BioSeqBench.models.utr5_mpra import decode_utr_one_hot, encode_utr_sequences


def test_utr5_encoding_round_trip():
    sequences = np.asarray(["ACGT" * 12 + "AC", "T" * 50])
    encoded = encode_utr_sequences(sequences)
    assert encoded.shape == (2, 4, 50)
    assert np.allclose(encoded.sum(dim=1).numpy(), 1)
    assert decode_utr_one_hot(encoded).tolist() == sequences.tolist()
    with pytest.raises(ValueError, match="50-nt"):
        encode_utr_sequences(["ACGT"])


def test_shared_utr5_initialization_is_unique_and_unconditioned():
    visible = np.asarray(["A" * 49 + base for base in "ACGT"] * 30)
    # Supply unique visible strings while retaining a simple deterministic tail.
    visible = benchmark._random_sequences(200, np.random.default_rng(4))
    config = benchmark.UTR5OptimizationConfig(num_replicates=1)
    initial, sources = benchmark._shared_initial_sequences(
        visible, config, np.random.default_rng(7)
    )
    assert len(initial) == 100
    assert len(set(initial.tolist())) == 100
    assert sources.tolist() == ["train_top"] * 50 + ["full_space_random"] * 50
    assert initial[:50].tolist() == visible[-50:][::-1].tolist()


def test_utr5_candidate_archive_excludes_visible_and_ranks_scores():
    archive = benchmark.CandidateArchive()
    sequences = ["A" * 50, "C" * 50, "G" * 50]
    archive.add(sequences, [1.0, 3.0, 2.0], [0.0, 0.0, 0.0], step=0)
    selected, scores, _ = archive.ranked({"C" * 50}, 2)
    assert selected.tolist() == ["G" * 50, "A" * 50]
    assert scores.tolist() == [2.0, 1.0]


def test_utr5_config_enforces_shared_seed_and_budget_invariants():
    benchmark.UTR5OptimizationConfig().validate()
    with pytest.raises(ValueError, match="shared initialization"):
        benchmark.UTR5OptimizationConfig(
            top_visible_initializations=49, random_initializations=50
        ).validate()
    with pytest.raises(ValueError, match="divisible"):
        benchmark.UTR5OptimizationConfig(query_budget=10_001).validate()


def test_saturation_tracker_selects_first_archive_at_90_percent_of_gain():
    config = saturation.SaturationConfig(
        train_fractions=(0.01,), minimum_queries=100,
        plateau_patience_queries=100, maximum_queries=1_000,
        material_improvement=1e-4,
    )
    tracker = saturation.SaturationTracker(set(), config)
    archive = benchmark.CandidateArchive()
    translation = str.maketrans("0123", "ACGT")
    sequences = np.asarray([f"{index:050d}".translate(translation) for index in range(100)])
    for query_count, score in [(100, 0.0), (200, 0.5), (300, 0.9), (400, 0.99), (500, 1.0)]:
        archive.add(sequences, np.full(100, score), np.zeros(100), step=query_count)
        tracker.observe(query_count, archive)
    result = tracker.finalize("synthetic", 600, "post_knee_plateau", {})
    assert result.q90_query_count == 300
    assert result.q90_top100_mean == pytest.approx(0.9)
    assert result.saturation_top100_mean == pytest.approx(1.0)
    assert result.result.query_count == 300
    assert result.trajectory["is_q90"].sum() == 1
    assert result.trajectory["is_knee"].sum() == 1


def test_saturation_config_validates_query_and_initialization_limits():
    saturation.SaturationConfig().validate()
    with pytest.raises(ValueError, match="initialization"):
        saturation.SaturationConfig(top_visible_initializations=49).validate()
    with pytest.raises(ValueError, match="minimum/maximum"):
        saturation.SaturationConfig(minimum_queries=10, maximum_queries=5).validate()
