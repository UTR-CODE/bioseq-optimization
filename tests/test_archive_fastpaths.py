import numpy as np
import pytest

from sequence_optimization import ScoreBatch
from sequence_optimization.advanced import _TopArchiveTracker
from sequence_optimization.algorithms import _select_diverse
from sequence_optimization.saturation import _SaturationMonitor, _SaturationReached


def test_incremental_monitor_matches_full_sort_with_updates_and_reset():
    rng = np.random.default_rng(71)
    monitor = _SaturationMonitor(10, 100000, 100000, 1e-4, 3)
    archive = {}
    for iteration in range(30):
        if iteration == 15:
            monitor.reset(iteration * 30)
            archive.clear()
        sequences = [''.join(s) for s in rng.choice(list('ACGT'), (30, 8))]
        if archive:
            sequences[:5] = list(archive)[:5]
        scores = rng.integers(0, 10, size=30).astype(float)
        for s, score in zip(sequences, scores):
            if s not in archive or score > archive[s][0]:
                archive[s] = (score, 0.)
        monitor.observe(np.asarray(sequences), ScoreBatch(scores), iteration * 30)
        expected = _select_diverse(sorted(archive.items(), key=lambda item: (-item[1][0], item[0])), 10, 3)
        assert monitor.snapshots[-1][1].tolist() == [s for s, _ in expected]
        assert monitor.ranking == sorted((-v[0], k) for k, v in archive.items())


def test_incomplete_prefix_does_not_apply_fallback():
    ranked = [('AA', (3., 0.)), ('AC', (2., 0.)), ('CA', (1., 0.))]
    assert len(_select_diverse(ranked, 2, 2, fallback=False)) == 1
    monitor = _SaturationMonitor(2, 1000, 1000, 1e-4, 2)
    monitor.observe(np.array(['AA', 'AC', 'CA']), ScoreBatch([3., 2., 1.]), 3)
    assert monitor.snapshots[-1][1].tolist() == ['AC', 'CA']


def test_cached_selection_still_advances_plateau_and_handles_ties():
    monitor = _SaturationMonitor(2, 2, 3, 1e-4, 1)
    monitor.observe(np.array(['CC', 'GG']), ScoreBatch([2., 1.]), 2)
    first = monitor._selection
    monitor.observe(np.array(['TT']), ScoreBatch([0.]), 3)
    assert monitor._selection is first
    # Equal score with lexicographically earlier key must invalidate selection.
    monitor.observe(np.array(['AA']), ScoreBatch([1.]), 4)
    assert monitor.snapshots[-1][1].tolist() == ['CC', 'AA']
    with pytest.raises(_SaturationReached):
        monitor.observe(np.array(['TT']), ScoreBatch([0.]), 5)


def test_high_scoring_candidate_rejected_by_selected_predecessor_uses_cache():
    monitor = _SaturationMonitor(2, 1000, 1000, 1e-4, 2)
    monitor.observe(np.array(["AA", "CC"]), ScoreBatch([3.0, 2.0]), 2)
    first = monitor._selection

    # AC ranks ahead of CC, but AA rejects it at diversity=2.  The exact
    # greedy selection therefore cannot change and need not be recomputed.
    monitor.observe(np.array(["AC"]), ScoreBatch([2.5]), 3)

    assert monitor._selection is first
    assert monitor.snapshots[-1][1].tolist() == ["AA", "CC"]
    assert monitor.ranking == [(-3.0, "AA"), (-2.5, "AC"), (-2.0, "CC")]


def test_continuous_subthreshold_gains_do_not_accumulate_into_progress():
    monitor = _SaturationMonitor(1, 1, 3, 1e-2, 0)
    monitor.observe(np.array(["A"]), ScoreBatch([0.0]), 1)
    monitor.observe(np.array(["C"]), ScoreBatch([0.004]), 2)
    monitor.observe(np.array(["G"]), ScoreBatch([0.008]), 3)
    with pytest.raises(_SaturationReached):
        monitor.observe(np.array(["T"]), ScoreBatch([0.012]), 4)


def test_single_dynamic_material_gain_refreshes_patience():
    monitor = _SaturationMonitor(1, 1, 3, 1e-2, 0)
    monitor.observe(np.array(["A"]), ScoreBatch([0.0]), 1)
    monitor.observe(np.array(["C"]), ScoreBatch([0.02]), 2)
    monitor.observe(np.array(["G"]), ScoreBatch([0.024]), 3)
    monitor.observe(np.array(["T"]), ScoreBatch([0.028]), 4)
    with pytest.raises(_SaturationReached):
        monitor.observe(np.array(["AA"]), ScoreBatch([0.032]), 5)


def test_bounded_top_archive_tracker_matches_full_greedy_selection():
    rng = np.random.default_rng(19)
    sequences = ["".join(row) for row in rng.choice(list("ACGT"), (600, 24))]
    archive = {
        sequence: (float(score), 0.0)
        for sequence, score in zip(sequences[:150], rng.normal(size=150))
    }
    tracker = _TopArchiveTracker(archive, count=20, diversity=5)

    for start in range(150, 600, 75):
        additions = sequences[start : start + 75]
        archive.update(
            (sequence, (float(score), 0.0))
            for sequence, score in zip(additions, rng.normal(size=len(additions)))
        )
        tracker.update(additions)
        selected, scores = tracker.select()
        expected = _select_diverse(
            sorted(archive.items(), key=lambda item: (-item[1][0], item[0])),
            20,
            5,
        )
        assert selected.tolist() == [item[0] for item in expected]
        np.testing.assert_array_equal(scores, [item[1][0] for item in expected])
