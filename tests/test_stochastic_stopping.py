import numpy as np
import pytest

from sequence_optimization import CallableScorer, SaturationConfig
from sequence_optimization.algorithms import optimize


@pytest.mark.parametrize('algorithm', ['simulated_annealing', 'cross_entropy_method'])
@pytest.mark.parametrize('plateau', [False, True])
def test_cached_batches_do_not_stop_stochastic_search(algorithm, plateau):
    calls = []
    def score(sequences):
        calls.extend(sequences)
        return np.zeros(len(sequences))
    options = dict(saturation=SaturationConfig(minimum_queries=4,
        plateau_patience_queries=8)) if plateau else dict(budget=20)
    result = optimize(CallableScorer(score), 1, ['A','C','G','T'], ['maximize'], 4,
        algorithm=algorithm, diversity=0, count_repeated=False,
        cross_entropy_batch_size=4, **options)
    if plateau:
        assert result.stop_reason == 'plateau'
        assert result.total_query_count == 12
    else:
        assert result.query_count == 20
    # All later proposals were duplicates, but no extra inference was needed.
    assert len(calls) == 4


def test_beam_parent_selection_obeys_diversity(monkeypatch):
    import sequence_optimization.advanced as advanced
    original = advanced._TopArchiveTracker.select
    parents = []

    def capture(tracker):
        values, scores = original(tracker)
        parents.append((values.copy(), tracker.diversity))
        return values, scores

    monkeypatch.setattr(advanced._TopArchiveTracker, "select", capture)
    result = optimize(CallableScorer(lambda s: [x.count('G') for x in s]), 6,
        ['GGGGGG','GGGGGA','AAAAAA','CCCCCC'], ['maximize'], 4,
        algorithm='beam_search', diversity=2, elite_fraction=.5, budget=80)
    assert len(parents) >= 2
    for values, distance in parents:
        assert distance == 2
        assert all(sum(a != b for a,b in zip(left,right)) >= 2
            for i,left in enumerate(values) for right in values[i+1:])
    assert result.metadata['parent_diversity'] == 2
