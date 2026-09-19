import numpy as np
import pytest

from sequence_optimization.algorithms import _diverse_parent_subset
from sequence_optimization.api import OptimizationConfig
from sequence_optimization import CallableScorer
from sequence_optimization.algorithms import optimize


def test_parent_selection_keeps_best_and_explores_other_sequence_families():
    sequences = np.array(['AAAAAAAA', 'AAAAAAAC', 'AAAAAACC', 'CCCCCCCC', 'GGGGGGGG'])
    scores = np.array([10., 9., 8., 7., 6.])
    parents, values = _diverse_parent_subset(sequences, scores, 3)
    assert parents.tolist() == ['AAAAAAAA', 'CCCCCCCC', 'GGGGGGGG']
    assert values.tolist() == [10., 7., 6.]
    reversed_parents, _ = _diverse_parent_subset(sequences[::-1], scores[::-1], 3)
    np.testing.assert_array_equal(parents, reversed_parents)


def test_cluster_cap_and_rotation():
    from sequence_optimization.algorithms import _cluster_parent_subset
    close = ['C' * i + 'A' + 'C' * (49-i) for i in range(15)]
    sequences = np.array(close + ['G' * 50, 'T' * 50])
    scores = np.arange(len(sequences), 0, -1, dtype=float)
    parents, _ = _cluster_parent_subset(sequences, scores, 100)
    assert len(parents) == 12
    assert sum(s in close for s in parents) == 10
    assert parents[:3].tolist() == [close[0], 'G' * 50, 'T' * 50]
    rotated, _ = _cluster_parent_subset(sequences, scores, 1, offset=1)
    assert rotated.tolist() == ['G' * 50]


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_parent_pool_multiplier(value):
    with pytest.raises(ValueError, match='parent_pool_multiplier'):
        OptimizationConfig(budget=100, parent_pool_multiplier=value)


@pytest.mark.parametrize('algorithm', ['adalead', 'beam_search', 'genetic_algorithm'])
def test_algorithms_expand_parent_pool_but_preserve_parent_count(monkeypatch, algorithm):
    import sequence_optimization.algorithms as algorithms
    import sequence_optimization.advanced as advanced
    module = advanced if algorithm == 'beam_search' else algorithms
    helper = '_diverse_parent_subset' if algorithm == 'genetic_algorithm' else '_cluster_parent_subset'
    original = getattr(module, helper)
    observed = []

    def capture(sequences, scores, count, **kwargs):
        selected, values = original(sequences, scores, count, **kwargs)
        observed.append((len(sequences), count, len(selected)))
        return selected, values

    monkeypatch.setattr(module, helper, capture)
    result = optimize(
        CallableScorer(lambda seqs: [s.count('G') for s in seqs]), 8,
        ['AAAAAAAA', 'CCCCCCCC', 'GGGGGGGG', 'TTTTTTTT'], ['maximize'], 4,
        algorithm=algorithm, elite_fraction=0.5, parent_pool_multiplier=2,
        diversity=0, budget=40,
    )
    assert observed[0] == (4, 2, 2)
    assert all(pool <= 2 * count and actual <= count for pool, count, actual in observed)
    assert result.metadata['parent_pool_multiplier'] == 2
