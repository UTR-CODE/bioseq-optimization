import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sequence_optimization.advanced import _nearest_neighbor_similarity
from sequence_optimization.algorithms import _cluster_parent_subset, optimize
from sequence_optimization import CallableScorer


def test_den_detects_duplicates_missed_by_fixed_pairs():
    # Old pairing A-C and A-G misses the duplicate A-A pair.
    x = F.one_hot(torch.tensor([[0]*5, [0]*5, [1]*5, [2]*5]), 4).float().requires_grad_()
    loss = _nearest_neighbor_similarity(x)
    assert loss.item() == pytest.approx(0.5)
    assert _nearest_neighbor_similarity(x[[2, 0, 3, 1]]).item() == pytest.approx(0.5)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize('count', [1, 3, 5])
def test_den_neighbor_edge_cases(count):
    x = torch.ones(count, 6, 4, requires_grad=True) / 4
    loss = _nearest_neighbor_similarity(x)
    assert loss.item() == pytest.approx(0 if count == 1 else .25)
    loss.backward()


def test_survival_preserves_low_score_family_and_unique_population():
    close = ['A'*i + 'C' + 'A'*(19-i) for i in range(8)]
    seqs = np.array(close + ['G'*20, 'T'*20, close[0]])
    scores = np.array(list(range(20, 12, -1)) + [2, 1, 20])
    selected, _ = _cluster_parent_subset(seqs, scores, 6, cap=6)
    assert len(set(selected)) == 6
    assert 'G'*20 in selected and 'T'*20 in selected


def test_adalead_rejects_child_better_than_slot_but_worse_than_parent(monkeypatch):
    import sequence_optimization.algorithms as module
    from sequence_optimization.api import SequenceSpace

    observed = []
    original = module._cluster_parent_subset

    def capture(sequences, scores, count, **kwargs):
        if kwargs.get('cap') == 2:
            observed.append(list(sequences))
        return original(sequences, scores, count, **kwargs)

    monkeypatch.setattr(module, '_cluster_parent_subset', capture)
    monkeypatch.setattr(SequenceSpace, 'mutate_one', lambda self, parents, rng: np.array(['AAAC', 'AAAG']))
    values = {'AAAA': 10., 'CCCC': 0., 'AAAC': 9., 'AAAG': 8.}
    result = optimize(CallableScorer(lambda seqs: [values[s] for s in seqs]),
                      4, ['AAAA', 'CCCC'], ['maximize'], 2,
                      algorithm='adalead', budget=4, diversity=0, elite_fraction=.5)
    assert observed == [['AAAA', 'CCCC']]
    assert result.metadata['offspring_comparison'] == 'own_parent'
    assert result.metadata['survivor_selection'] == 'family_round_robin'
