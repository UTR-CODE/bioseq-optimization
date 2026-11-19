import numpy as np
import pytest

from sequence_optimization import (
    CallableScorer,
    SaturationConfig,
    SequencePredictor,
    available_algorithms,
)
from sequence_optimization.algorithms import (
    _hamming_distance,
    _score,
    _select_diverse,
    optimize,
)
from sequence_optimization.saturation import (
    _SaturationMonitor,
    _blend_to_mean_floor,
    _knee_index,
    _quality_diverse_snapshot,
)


@pytest.mark.parametrize(
    "means,expected",
    [
        ([0, 8, 9, 9, 9], 1),
        ([2, 2, 2, 2, 2], 0),
        ([0, 1, 2, 3, 4], 4),
        ([0, 1, 3, 6, 10], 4),
    ],
)
def test_knee_curve_shapes(means, expected):
    assert _knee_index(np.arange(5), np.array(means, dtype=float), 1e-4) == expected


def test_no_implicit_ceiling_and_real_knee_archive(monkeypatch):
    from sequence_optimization.algorithms import _REGISTRY

    def scripted(initial, scorer, space, config, rng):
        assert scorer.budget is None
        archive = {}
        for _ in range(4):
            _score(scorer, archive, initial)

    monkeypatch.setitem(_REGISTRY, "knee_test", scripted)
    r = optimize(
        CallableScorer(lambda seqs: np.zeros(len(seqs))),
        1,
        ["A"],
        ["maximize"],
        1,
        algorithm="knee_test",
        diversity=0,
        saturation=SaturationConfig(minimum_queries=1, plateau_patience_queries=100001),
        count_repeated=True,
    )
    assert r.knee.query_count == r.knee_query_count
    assert r.selected is r.at(0.9)
    assert r.selected.metadata["saturation_fraction"] == 0.9
    assert r.knee.metadata["maximum_queries"] is None
    assert not r.knee.metadata["plateau_confirmed"]


def test_selects_earliest_archive_at_ninety_percent_of_plateau_gain(monkeypatch):
    from itertools import product
    from sequence_optimization.algorithms import _REGISTRY

    seqs = ["".join(s) for s in product("ACGT", repeat=3)][:30]
    scores = dict(zip(seqs, [0.0, 7.0, 8.2, 9.0] + [9.0] * 26))

    def scripted(initial, scorer, space, config, rng):
        archive = {}
        for seq in seqs:
            _score(scorer, archive, np.array([seq]))

    monkeypatch.setitem(_REGISTRY, "knee_curve", scripted)
    r = optimize(
        CallableScorer(lambda s: [scores[x] for x in s]),
        3,
        [seqs[0]],
        ["maximize"],
        1,
        algorithm="knee_curve",
        diversity=0,
        saturation=SaturationConfig(
            minimum_queries=1,
            plateau_patience_queries=100,
            relaxed_knee=False,
            knee_fraction=0.9,
        ),
    )
    assert r.knee.query_count == 4
    assert r.selected.query_count == 3
    assert r.selected.scores[0] == 8.2
    assert r.selected.metadata["target_top_mean"] == pytest.approx(8.1)
    assert r.total_query_count == 30


def test_relaxed_quality_band_expands_without_lowering_set_mean_floor():
    sequences = np.asarray(["AAAA", "CCCC", "GGGG", "TTTT"])
    scores = np.asarray([1.0, 0.95, 0.9, 0.8])
    monitor = _SaturationMonitor(3, 1, 10, 1e-2, 0)
    monitor.archive = {
        sequence: (float(score), 0.0)
        for sequence, score in zip(sequences, scores)
    }
    monitor.discovery_queries = {
        sequence: index + 1 for index, sequence in enumerate(sequences)
    }
    fallback = (4, sequences[:3], scores[:3], np.zeros(3))

    snapshot, metadata = _quality_diverse_snapshot(
        monitor,
        count=3,
        baseline=0.0,
        target=1.0,
        knee_gain=1.0,
        tolerance=0.01,
        minimum_distance=0,
        pool_size=100,
        seed=3,
        fallback=fallback,
        mean_score_floor=0.85,
    )

    assert metadata["quality_band_expanded"]
    assert metadata["selection_mean_floor_satisfied"]
    assert snapshot[2].mean() >= 0.85


def test_mean_floor_blend_replaces_the_minimum_number_of_diverse_candidates():
    keys = np.asarray(["AAAA", "CCCC", "GGGG", "AAAC", "CCCG", "GGGT"])
    scores = np.asarray([0.4, 0.5, 0.6, 1.0, 0.9, 0.8])
    selected, replacements = _blend_to_mean_floor(
        keys,
        scores,
        np.asarray([0, 1, 2]),
        keys[[3, 4, 5]],
        mean_floor=0.7,
        minimum_distance=0,
    )

    assert replacements == 1
    assert set(selected) == {1, 2, 3}
    assert scores[selected].mean() >= 0.7


def test_plateau_gain_applies_to_every_nonrandom_algorithm():
    torch = pytest.importorskip("torch")

    class SmoothScore(torch.nn.Module):
        def forward(self, values):
            weights = values.new_tensor([0.0, 0.2, 1.0, -0.1])
            return (values * weights[None, :, None]).sum((1, 2)).unsqueeze(1)

    model = SequencePredictor(SmoothScore(), device="cpu")
    initial = np.asarray(
        [
            "AAAAAAAA",
            "CCCCCCCC",
            "GGGGGGGG",
            "TTTTTTTT",
            "ACGTACGT",
            "TGCATGCA",
            "AGAGAGAG",
            "CTCTCTCT",
        ]
    )
    algorithms = set(available_algorithms()) - {"random"}

    for algorithm in sorted(algorithms):
        result = optimize(
            model,
            8,
            initial,
            ["maximize"],
            4,
            algorithm=algorithm,
            diversity=1,
            seed=7,
            saturation=SaturationConfig(
                minimum_queries=8,
                plateau_patience_queries=16,
            ),
            generator_pretrain_epochs=1,
            generator_update_epochs=1,
            den_seed_pretrain_epochs=1,
            cross_entropy_batch_size=8,
            basesearch_batch_size=8,
        )
        metadata = result.selected.metadata
        baseline = metadata["baseline_top_mean"]
        plateau = metadata["plateau_top_mean"]
        assert metadata["target_top_mean"] == pytest.approx(baseline + 0.9 * (plateau - baseline))
        assert len(set(result.selected.sequences)) == 4
        if plateau >= baseline:
            assert result.selected.scores.mean() >= metadata["target_top_mean"] - 1e-9


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"knee_minimum_gain_fraction": 0.0}, "knee_minimum"),
        ({"selection_minimum_gain_fraction": -0.1}, "selection_minimum"),
        (
            {
                "knee_minimum_gain_fraction": 0.8,
                "selection_minimum_gain_fraction": 0.9,
            },
            "cannot exceed",
        ),
    ],
)
def test_relaxed_knee_fraction_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SaturationConfig(**kwargs).resolve(4)


def test_quality_diverse_selection_prefers_spread_inside_target_band():
    clustered = ["AAAAAAAA", "AAAAAAAC", "AAAAAACA", "AAAAACAA"]
    spread = ["CCCCCCCC", "GGGGGGGG", "TTTTTTTT", "ACGTACGT"]
    sequences = clustered + spread
    scores = np.asarray([0.99, 0.98, 0.97, 0.96, 0.91, 0.90, 0.89, 0.88])
    monitor = _SaturationMonitor(4, 1, 10, 1e-2, 0)
    monitor.archive = {
        sequence: (float(score), 0.0) for sequence, score in zip(sequences, scores)
    }
    monitor.discovery_queries = {
        sequence: index + 1 for index, sequence in enumerate(sequences)
    }
    fallback_items = _select_diverse(
        sorted(monitor.archive.items(), key=lambda item: (-item[1][0], item[0])),
        4,
        0,
    )
    fallback = (
        4,
        np.asarray([item[0] for item in fallback_items]),
        np.asarray([item[1][0] for item in fallback_items]),
        np.zeros(4),
    )

    snapshot, metadata = _quality_diverse_snapshot(
        monitor,
        count=4,
        baseline=0.0,
        target=0.9,
        knee_gain=1.0,
        tolerance=0.1,
        minimum_distance=0,
        pool_size=100,
        seed=3,
        fallback=fallback,
    )

    old_minimum = min(
        _hamming_distance(left, right)
        for index, left in enumerate(fallback[1])
        for right in fallback[1][index + 1 :]
    )
    assert metadata["minimum_pairwise_hamming"] > old_minimum
    assert metadata["minimum_pairwise_hamming"] >= 6
    assert snapshot[2].min() >= 0.8
    assert snapshot[2].max() <= 1.0
    assert not metadata["quality_diversity_fallback"]
