"""Behavioral checks for quality-gated motif/position cooperative search."""

import itertools

import numpy as np
import pandas as pd
import pytest

from sequence_optimization import (
    CallableScorer,
    DiversitySearchConfig,
    MotifPositionDescriptor,
    MultiTaskObjective,
    generate_designs,
    random_seed_sequences,
    available_algorithms,
)
from sequence_optimization.api import ScoreBatch
from sequence_optimization.exploration import _PatternArchive, generate_diverse_designs


def test_descriptor_merges_identity_but_can_preserve_orientation_and_copy_count():
    descriptor = MotifPositionDescriptor({"GATA": ["AGATAAG", "CTTATCT"]}, n_bins=2)
    np.testing.assert_array_equal(
        descriptor(["AGATAAGAAAAAAA", "CTTATCTAAAAAAA"]), [[1, 0], [1, 0]]
    )
    oriented = MotifPositionDescriptor(
        {"GATA": ["AGATAAG", "CTTATCT"]}, n_bins=2, include_strand=True
    )
    np.testing.assert_array_equal(
        oriented(["AGATAAGAAAAAAA", "CTTATCTAAAAAAA"]), [[1, 0, 0, 0], [0, 1, 0, 0]]
    )
    # Multiple hits are counts, not multiple motifs or independent hit windows.
    count = MotifPositionDescriptor({"A": ["AA"]}, n_bins=1, max_count=2)
    assert count(["AAAA"]).tolist() == [[2]]
    palindrome = MotifPositionDescriptor({"pal": ["AT"]}, n_bins=1, include_strand=True)
    assert palindrome(["AT"]).tolist() == [[1, 0]]


def test_descriptor_is_independent_of_batch_and_rejects_duplicate_motif_identities():
    descriptor = MotifPositionDescriptor({"x": ["AC"]}, n_bins=3)
    seqs = ["ACAAAA", "AAAAGT"]
    np.testing.assert_array_equal(
        descriptor(seqs), np.concatenate([descriptor([s]) for s in seqs])
    )
    with pytest.raises(ValueError, match="different motifs"):
        MotifPositionDescriptor({"a": ["AC"], "b": ["GT"]})


def test_targeted_proposals_respect_fixed_regions_and_fill_missing_positions():
    descriptor = MotifPositionDescriptor({"x": ["GG"]}, n_bins=2)
    proposals = descriptor.propose(
        ["AAAAAAAA"] * 40, [1000, 0], np.random.default_rng(42), (0, 1, 2, 3)
    )
    assert all(s.startswith("AAAA") and len(s) == 8 for s in proposals)
    assert descriptor(proposals)[:, 1].sum() == 40


def test_archive_rejects_low_quality_and_unresolved_patterns_and_prunes_after_improvement():
    descriptor = MotifPositionDescriptor({"x": ["GG"]}, n_bins=2)
    archive = _PatternArchive(
        descriptor,
        DiversitySearchConfig(quality_fraction=0.8),
        MultiTaskObjective(directions=["maximize"]),
        None,
    )
    archive.observe(
        ["GGAAAAAA", "AAAAGGAA"], ScoreBatch([0.0, 0.0]), "initial", initialize=True
    )
    archive.observe(
        ["GGAAAACC", "AAAAGGCC", "AAAAAAAA"], ScoreBatch([10.0, 7.0, 99.0]), "worker"
    )
    assert archive.floor == 8
    assert set(archive.entries) == {"GGAAAACC"}
    # A new high-quality position is retained; lower-quality novelty never fills a quota.
    archive.observe(["AAAAGGCC"], ScoreBatch([9.0]), "worker")
    assert len(archive.select(10)) == 2
    assert len(archive.select(10, min_hamming=8)) == 1


def test_archive_admits_quality_bounded_candidates_even_if_initial_seeds_are_below_bounds():
    config = DiversitySearchConfig(
        activity_bounds={"K562": (5, None), "off": (None, 1)},
        quality_fn=lambda x: x[:, 0] - x[:, 1],
        quality_fraction=0,
    )
    archive = _PatternArchive(
        MotifPositionDescriptor({"x": ["GG"]}, n_bins=2),
        config,
        MultiTaskObjective(directions=["maximize", "minimize"]),
        ("K562", "off"),
    )
    archive.observe(["GGAAAAAA"], ScoreBatch([[1.0, 0.0]]), "initial", initialize=True)
    assert not archive.entries
    archive.observe(
        ["GGAAAACC", "AAAAGGCC"], ScoreBatch([[6.0, 0.0], [10.0, 3.0]]), "worker"
    )
    assert set(archive.entries) == {"GGAAAACC"}
    assert archive.entries["GGAAAACC"]["quality"] == 6


def _search(**kwargs):
    args = dict(
        model=CallableScorer(lambda s: np.array([x.count("G") for x in s], float)),
        length=12,
        initial=random_seed_sequences(12, 8, seed=2),
        objective=["maximize"],
        descriptor=MotifPositionDescriptor(
            {"GATA": ["AGATAAG"], "GC": ["GGCC"]}, n_bins=4
        ),
        config=DiversitySearchConfig(
            total_budget=512, segment_budget=64, quality_fraction=0.5
        ),
        algorithms=["genetic_algorithm", "random"],
        num_candidates=5,
        diversity=2,
        seed=7,
    )
    args.update(kwargs)
    return generate_diverse_designs(**args)


def test_search_budget_global_spacing_quality_and_exchange_are_reproducible():
    result = _search()
    again = _search()
    pd.testing.assert_frame_equal(result.candidates, again.candidates)
    pd.testing.assert_frame_equal(result.history, again.history)
    assert result.metadata["query_count"] <= 512
    assert result.metadata["query_count"] == 8 + result.history.segment_queries.sum()
    assert result.metadata["query_count"] == 8 + sum(
        result.metadata["per_algorithm_queries"].values()
    )
    assert result.metadata["unique_model_queries"] <= result.metadata["query_count"]
    assert set(result.history.algorithm) == {"genetic_algorithm", "random"}
    assert result.history.exchanged_seeds.sum() > 0
    assert len(result.candidates) == 5
    assert result.candidates.quality.ge(result.metadata["quality_floor"]).all()
    assert result.candidates.sequence.is_unique
    for a, b in itertools.combinations(result.candidates.sequence, 2):
        assert sum(x != y for x, y in zip(a, b)) >= 2


def test_hard_constraints_fixed_regions_and_shared_cache_apply_to_all_proposals():
    queried = []

    def score(seqs):
        queried.extend(seqs)
        assert all(s.startswith("AA") and "TTT" not in s for s in seqs)
        return np.array([s.count("G") for s in seqs], float)

    seeds = ["AA" + s for s in random_seed_sequences(10, 20, seed=2) if "TTT" not in s][
        :8
    ]
    result = _search(
        model=CallableScorer(score),
        initial=seeds,
        fixed_region=[0, 1],
        constraint=lambda seqs: np.array(["TTT" not in s for s in seqs], float),
    )
    assert len(queried) == len(set(queried)) == result.metadata["unique_model_queries"]
    assert result.candidates.sequence.str.startswith("AA").all()
    assert not result.candidates.sequence.str.contains("TTT").any()


def test_all_scored_candidates_enter_archive_even_when_local_optimizer_does_not_return_them(
    monkeypatch,
):
    import sequence_optimization.algorithms as algorithms
    from sequence_optimization.api import OptimizationResult

    def optimize(model, length, initial, objective, num_candidates, **kwargs):
        # This excellent sequence is scored but omitted from the local returned panel.
        model.score(np.array(["GGCCGGCCGGCC"]))
        return OptimizationResult(
            kwargs["algorithm"],
            np.array(initial),
            np.zeros(len(initial)),
            np.zeros(len(initial)),
            len(initial) + 1,
            {},
            [],
        )

    monkeypatch.setattr(algorithms, "optimize", optimize)
    result = _search(
        config=DiversitySearchConfig(
            total_budget=128,
            segment_budget=64,
            quality_fraction=0,
            motif_proposal_fraction=0,
        )
    )
    assert "GGCCGGCCGGCC" in set(result.archive.sequence)


def test_empty_archive_reports_shortfall_and_can_be_saved(tmp_path):
    result = _search(
        config=DiversitySearchConfig(
            total_budget=128, segment_budget=64, min_quality=100
        )
    )
    assert result.candidates.empty
    assert result.metadata["shortfall"] == 5
    result.save(tmp_path)
    assert (tmp_path / "metadata.json").exists()
    assert pd.read_csv(tmp_path / "candidates.csv").empty


def test_ablation_disables_exchange_and_directed_proposals():
    result = _search(
        config=DiversitySearchConfig(
            total_budget=256,
            segment_budget=64,
            exchange_seeds=False,
            motif_proposal_fraction=0,
        )
    )
    assert result.history.motif_proposals.sum() == 0
    assert result.history.exchanged_seeds.sum() == 0


def test_gradient_search_tracks_soft_work_and_uses_authoritative_hard_model():
    import torch

    class Gradient:
        def score_soft(self, p):
            return p[:, :, 2].sum(1), None

    result = _search(
        algorithms=["fast_seqprop"],
        differentiable_model=Gradient(),
        config=DiversitySearchConfig(
            total_budget=128, segment_budget=64, quality_fraction=0.2
        ),
    )
    assert result.metadata["soft_sequence_evaluations"] > 0
    assert result.metadata["query_count"] <= 128
    for row in result.archive.itertuples():
        assert row.activity_task_0 == row.sequence.count("G")
    assert torch.isfinite(torch.tensor(result.archive.score.to_numpy())).all()


def test_suite_opt_in_returns_global_panel_and_metadata():
    frame = generate_designs(
        CallableScorer(lambda s: np.array([x.count("G") for x in s])),
        12,
        random_seed_sequences(12, 8, seed=2),
        ["maximize"],
        algorithms=["random"],
        num_candidates=3,
        diversity=1,
        exploration=DiversitySearchConfig(total_budget=128, segment_budget=64),
        descriptor=MotifPositionDescriptor({"GC": ["GGCC"]}, n_bins=4),
    )
    assert len(frame) <= 3  # A global panel, not N rows per algorithm.
    assert frame.attrs["diversity_search"]["query_count"] <= 128


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"config": DiversitySearchConfig(total_budget=16, segment_budget=64)},
            "total_budget",
        ),
        (
            {"config": DiversitySearchConfig(total_budget=128, segment_budget=8)},
            "segment_budget",
        ),
        ({"algorithms": ["unknown"]}, "registered"),
        ({"algorithms": ["fast_seqprop"]}, "score_soft"),
        ({"algorithm_options": {"random": {"budget": 100}}}, "override"),
    ],
)
def test_invalid_search_rejected_before_inference(kwargs, match):
    def fail(_):
        raise AssertionError("model should not be called")

    with pytest.raises((ValueError, TypeError), match=match):
        _search(model=CallableScorer(fail), **kwargs)


def test_objective_bounds_are_never_relaxed():
    result = _search(
        objective=MultiTaskObjective(directions=["maximize"], stop_values=[6]),
        initial=["GGAAAAAAAACC", "AAAAGGAAAACC"],
        algorithms=["random"],
        config=DiversitySearchConfig(
            total_budget=96, segment_budget=32, quality_fraction=0
        ),
    )
    assert result.archive.activity_task_0.le(6).all()
    assert result.candidates.activity_task_0.le(6).all()


def test_seed_feedback_visits_only_committed_seeds():
    archive = _PatternArchive(
        MotifPositionDescriptor({"x": ["GG"]}, n_bins=2),
        DiversitySearchConfig(),
        MultiTaskObjective(directions=["maximize"]),
        None,
    )
    archive.observe(
        ["GGAAAAAA", "AAAAGGAA"], ScoreBatch([1.0, 1.0]), "initial", initialize=True
    )
    archive.select(2, feedback=True)
    assert not archive.visits


@pytest.mark.parametrize("algorithm", available_algorithms())
def test_controller_integrates_every_registered_optimizer(algorithm):
    import torch

    class Model:
        task_names = ("target", "off")

        def score(self, sequences):
            return np.array(
                [[s.count("G"), s.count("C")] for s in sequences], dtype=float
            )

        def score_soft(self, probabilities):
            return torch.stack(
                [probabilities[:, :, 2].sum(1), probabilities[:, :, 1].sum(1)], 1
            ), None

    result = generate_diverse_designs(
        Model(),
        8,
        ["GGAAAAAA", "AAAAGGAA", "GGAAAACC", "AAAAGGCC"],
        ["maximize", "minimize"],
        descriptor=MotifPositionDescriptor({"G": ["GG"]}, n_bins=2),
        config=DiversitySearchConfig(
            total_budget=96,
            segment_budget=48,
            quality_fraction=0,
            activity_bounds={"target": (1, None)},
        ),
        algorithms=[algorithm],
        num_candidates=3,
        diversity=1,
        algorithm_options={
            algorithm: dict(
                generator_pretrain_epochs=1,
                generator_update_epochs=1,
                den_seed_pretrain_epochs=1,
            )
        },
    )
    assert result.metadata["query_count"] <= 96
    assert len(result.candidates) > 0
    assert result.candidates.activity_target.ge(1).all()
    for row in result.candidates.itertuples():
        assert row.activity_target == row.sequence.count("G")
        assert row.activity_off == row.sequence.count("C")


def test_archive_capacity_is_bounded_without_relaxing_quality():
    descriptor = MotifPositionDescriptor({"G": ["G"]}, n_bins=2)
    archive = _PatternArchive(
        descriptor,
        DiversitySearchConfig(max_patterns=2, per_pattern=2, quality_fraction=0),
        MultiTaskObjective(directions=["maximize"]),
        None,
    )
    sequences = ["".join(s) for s in itertools.product("AG", repeat=4)]
    archive.observe(
        sequences, ScoreBatch(np.ones(len(sequences))), "initial", initialize=True
    )
    assert len(archive.entries) <= 4
    assert len({e["features"] for e in archive.entries.values()}) == 2
    assert all(e["quality"] >= archive.floor for e in archive.entries.values())
