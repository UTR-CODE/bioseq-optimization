import numpy as np
import pytest

from sequence_optimization import CallableScorer, ScoreBatch, generate_designs
from sequence_optimization.saturation import _SaturationMonitor


@pytest.mark.parametrize('entry', ['suite', 'direct'])
def test_rejects_smaller_population(entry):
    from sequence_optimization.algorithms import optimize
    function = generate_designs if entry == 'suite' else optimize
    options = {'algorithms': ['random']} if entry == 'suite' else {'algorithm': 'random'}
    with pytest.raises(ValueError, match='at least 10 sequences'):
        function(CallableScorer(lambda s: np.zeros(len(s))),
            8, ['AAAAAAAA', 'CCCCCCCC'], ['maximize'], num_candidates=10,
            score_threshold=0, diversity=0, **options)


def test_monitor_baseline_is_initial_not_first_complete_archive():
    m = _SaturationMonitor(3, 3, 30, 1e-4, 0)
    m.observe(np.array(['AA']), ScoreBatch([1.]), 1)
    assert m.initial_mean == 1.
    assert not m.snapshots
    m.observe(np.array(['AC', 'AG', 'AT']), ScoreBatch([5., 6., 7.]), 4)
    assert m.initial_mean == 1.
    assert len(m.snapshots[0][1]) == 3
    assert m.snapshots[0][2].mean() == 6.


def test_incomplete_collection_stalls_with_clear_error():
    m = _SaturationMonitor(3, 3, 3, 1e-4, 0)
    m.observe(np.array(['AA']), ScoreBatch([1.]), 1)
    with pytest.raises(RuntimeError, match='candidate collection stalled'):
        m.observe(np.array(['AA']), ScoreBatch([1.]), 4)
    assert not m.snapshots


def test_impossible_candidate_count_errors_before_search():
    with pytest.raises(ValueError, match='at least 4 sequences'):
        generate_designs(CallableScorer(lambda s: np.zeros(len(s))), 4,
            ['AAAA'], ['maximize'], algorithms=['simulated_annealing'],
            num_candidates=4, score_threshold=0, fixed_region=['0:4'], diversity=0)


def test_empty_initial_is_rejected():
    with pytest.raises(ValueError, match='at least one'):
        generate_designs(CallableScorer(lambda s: np.zeros(len(s))), 4, [],
            ['maximize'], algorithms=['random'], num_candidates=4)


def test_high_initial_mean_does_not_require_undersized_output(monkeypatch):
    from sequence_optimization.algorithms import _REGISTRY, optimize, _score
    from sequence_optimization import SaturationConfig
    def scripted(initial, scorer, space, config, rng):
        archive = {}
        _score(scorer, archive, initial)
        _score(scorer, archive, np.array(['CC', 'GG', 'TT']))
    monkeypatch.setitem(_REGISTRY, 'high_initial', scripted)
    r = optimize(CallableScorer(lambda s: [10. if x in ['AA','AC','AG'] else 1. for x in s]),
        2, ['AA','AC','AG'], ['maximize'], 3, algorithm='high_initial', diversity=2,
        saturation=SaturationConfig())
    assert len(r.selected.sequences) == 3
    assert r.selected.metadata['baseline_top_mean'] == 10.
    assert r.selected.metadata['plateau_top_mean'] <= r.selected.metadata['baseline_top_mean']
    assert r.selected.metadata['saturation_fraction'] == 0.9
