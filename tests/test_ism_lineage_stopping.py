import numpy as np
import pytest

from sequence_optimization import CallableScorer, OptimizationConfig, SaturationConfig, SequenceSpace
from sequence_optimization.algorithms import _ism_curve_stop, optimize
from sequence_optimization import api


def test_online_knee_requires_confirmed_slow_tail():
    cfg = OptimizationConfig(budget=100, ism_absolute_gain=0, ism_relative_gain=0)
    assert _ism_curve_stop([0, 10, 20, 30, 40], cfg)[0] is None
    reason, knee = _ism_curve_stop([0, 10, 20, 30, 40, 40.1, 40.2, 40.3, 40.4, 40.5, 40.6], cfg)
    assert reason == 'knee_slow_tail' and knee == 4
    assert _ism_curve_stop(list(range(30)), cfg)[0] is None
    assert _ism_curve_stop([i*i for i in range(30)], cfg)[0] is None


def test_small_gain_can_stop_while_still_improving():
    cfg = OptimizationConfig(budget=100)
    assert _ism_curve_stop([i * .001 for i in range(11)], cfg)[0] == 'small_gain'


def test_frozen_lineages_are_not_scanned_again():
    result = optimize(CallableScorer(lambda ss: [s.count('G') for s in ss]),
        6, ['GGGGGG', 'AAAAAA'], ['maximize'], 2,
        algorithm='ism_greedy', diversity=0, seed=42, count_repeated=True,
        saturation=SaturationConfig(minimum_queries=2))
    assert result.stop_reason == 'plateau'
    d = result.selected.metadata['ism_lineage_diagnostics']
    assert d['rounds'][0]['scanned_lineages'] == 2
    assert all(r['scanned_lineages'] == 1 for r in d['rounds'][1:])
    assert d['lineages'][0]['stop_sweep'] == 1
    assert d['lineages'][1]['stop_sweep'] == 7
    assert len(result.selected.sequences) == 2
    assert result.total_query_count == 2 + 8 * 18


def test_tiny_gain_lineage_stops_without_reaching_local_optimum():
    result = optimize(CallableScorer(lambda ss: [s.count('G') * .001 for s in ss]),
        20, ['A'*20], ['maximize'], 1, algorithm='ism_greedy', diversity=0,
        count_repeated=True, saturation=SaturationConfig(minimum_queries=1))
    d = result.selected.metadata['ism_lineage_diagnostics']
    assert d['lineages'][0]['stop_reason'] == 'small_gain'
    assert len(d['lineages'][0]['scores']) == 11
    assert d['lineages'][0]['scores'][-1] == pytest.approx(.01)


@pytest.mark.parametrize('alphabet,sequences', [('ACGT',['ACGT','TGCA']), ('AC',['ACAC']), ('TGCA',['TGCA'])])
def test_native_matches_python_order_and_values(monkeypatch, alphabet, sequences):
    if api._native_single_mutants is None:
        pytest.skip('optional C extension unavailable')
    space = SequenceSpace(4, alphabet)
    native = space.single_mutants(sequences)
    monkeypatch.setattr(api, '_native_single_mutants', None)
    np.testing.assert_array_equal(native, space.single_mutants(sequences))


def test_native_fixed_region_and_normalization_match_fallback(monkeypatch):
    from sequence_optimization.api import FixedRegionSequenceSpace
    # See constructor for fixed-position resolution; only positions 1 and 3 mutate.
    space = FixedRegionSequenceSpace(SequenceSpace(4), 'ACGT', (0, 2))
    native = space.single_mutants(['acgu'])
    monkeypatch.setattr(api, '_native_single_mutants', None)
    np.testing.assert_array_equal(native, space.single_mutants(['acgu']))
    assert all(s[0] == 'A' and s[2] == 'G' for s in native)


def test_native_rejects_invalid_boundary_inputs():
    native = pytest.importorskip('sequence_optimization._sequence_kernels')
    for values, alphabet, positions in [(['AN'], 'ACGT', [0]), (['AA'], 'ACGT', [-1]),
        (['AA'], 'ACGT', [2]), (['A', 'AA'], 'ACGT', [0]), (['AA'], 'AAC', [0])]:
        with pytest.raises(ValueError):
            native.single_mutants(values, alphabet, positions)
    assert native.single_mutants([], 'ACGT', [0]) == []


@pytest.mark.parametrize('kwargs', [dict(ism_patience=0), dict(ism_min_sweeps=True),
    dict(ism_absolute_gain=-1), dict(ism_relative_gain=float('nan')), dict(ism_knee_slope_ratio=2)])
def test_invalid_stopping_parameters(kwargs):
    with pytest.raises(ValueError, match='ism_'):
        OptimizationConfig(budget=100, **kwargs)


def test_knee_freezes_a_still_improving_lineage_in_search():
    def score(ss):
        counts = np.asarray([s.count('G') for s in ss])
        return 10 * np.minimum(counts, 4) + .1 * np.maximum(counts-4, 0)
    result = optimize(CallableScorer(score), 20, ['A'*20], ['maximize'], 1,
        algorithm='ism_greedy', diversity=0, count_repeated=True,
        ism_absolute_gain=0, ism_relative_gain=0,
        saturation=SaturationConfig(minimum_queries=1))
    item = result.selected.metadata['ism_lineage_diagnostics']['lineages'][0]
    assert item['stop_reason'] == 'knee_slow_tail'
    assert item['knee_sweep'] == 4 and item['stop_sweep'] == 10
    assert item['scores'][-1] > item['scores'][-2]


def test_stopping_can_be_disabled_for_control_runs():
    result = optimize(CallableScorer(lambda ss: np.zeros(len(ss))),
        4, ['AAAA','CCCC'], ['maximize'], 2, algorithm='ism_greedy', diversity=0,
        count_repeated=True, ism_lineage_stopping=False,
        saturation=SaturationConfig(minimum_queries=2))
    assert result.stop_reason == 'plateau'
    assert result.total_query_count == 2 + 20*2*4*3
    d = result.selected.metadata['ism_lineage_diagnostics']
    assert all(r['scanned_lineages'] == 2 for r in d['rounds'])
    assert all(item['stop_reason'] is None for item in d['lineages'])
