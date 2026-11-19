from tests.helpers import run_kernel

import numpy as np
import pytest
import torch

from sequence_optimization import CallableScorer, MultiTaskObjective, SaturationConfig
from sequence_optimization.objectives import ObjectiveScorer
from sequence_optimization.api import ScoreBatch
from sequence_optimization.saturation import _SaturationMonitor, _SaturationReached
from sequence_optimization.algorithms import available_algorithms


@pytest.mark.parametrize('aggregation', ['mean', 'worst', 'smoothmin'])
def test_per_candidate_maintenance_numpy_torch(aggregation):
    obj = MultiTaskObjective(directions=['maximize', 'minimize', 'minimize'],
                             stop_values=[(-2, 6)] * 3, aggregation=aggregation)
    x = torch.tensor([[5.8, -1.8, -1.7], [4., -1.8, -1.7],
                      [5.8, 0., 0.], [4., 0., 0.]], requires_grad=True)
    y = obj.scalarize_soft(x)
    np.testing.assert_allclose(y.detach(), obj.scalarize(x.detach().numpy()), atol=1e-6)
    y.sum().backward()
    assert torch.equal(x.grad[0], torch.zeros(3))
    assert x.grad[1, 0] > 0 and x.grad[1, 1:].eq(0).all()
    assert x.grad[2, 0] == 0 and x.grad[2, 1:].lt(0).all()
    assert x.grad[3, 0] > 0 and x.grad[3, 1:].lt(0).all()


def test_ready_candidate_does_not_deactivate_other_candidates():
    values = {'AAAA': [5.8, -1.8, -1.8], 'CCCC': [1., 0., 0.]}
    scorer = ObjectiveScorer(CallableScorer(lambda seqs: [values[s] for s in seqs]),
                             ['maximize', 'minimize', 'minimize'], stop_values=[6, -2, -2])
    before = scorer.score(['CCCC']).scores.copy()
    scorer.score(['AAAA'])
    np.testing.assert_array_equal(before, scorer.scores_for(['CCCC']))
    assert scorer.objective_version == 0
    assert not scorer.target_stop_reached
    assert scorer.scores_for(['AAAA'])[0] == 0
    # Reversion restores a penalty, rather than globally forgetting the task.
    values['AAAA'] = [4., -1.8, -1.8]
    assert scorer.score(['AAAA']).scores[0] == pytest.approx(-.5)


def test_maintenance_does_not_reward_movement_inside_band():
    obj = MultiTaskObjective(directions=['maximize','minimize','minimize'], stop_values=[6,-2,-2])
    x = np.array([[5.5,-1.5,-1.5],[6.,-2.,-2.]])
    np.testing.assert_array_equal(obj.scalarize(x), [0.,0.])
    assert obj.feasible(np.array([[6.01,-2.,-2.],[6.,-2.01,-2.]])).tolist() == [False,False]


def test_twenty_round_patience_independent_of_batch_query_counts():
    monitor = _SaturationMonitor(1,1,1,.01,0,0,patience_rounds=20)
    for r in range(1,21):
        monitor.observe(np.array(['AAAA']),ScoreBatch([0.]),r*10000)
    with pytest.raises(_SaturationReached):
        monitor.observe(np.array(['AAAA']),ScoreBatch([0.]),210000)


def test_accumulated_improvement_resets_round_patience():
    monitor = _SaturationMonitor(1,1,1,.01,0,0,patience_rounds=10)
    for r in range(30):
        monitor.observe(np.array(['AAAA']),ScoreBatch([r*.006]),r+1)
    assert monitor.observation_round - monitor.last_material_round < 2


@pytest.mark.parametrize('value', [-1, True, float('nan'), float('inf')])
def test_invalid_tolerance(value):
    with pytest.raises(ValueError, match='stop_tolerance'):
        MultiTaskObjective(directions=['maximize'],stop_values=[6],stop_tolerance=value)


@pytest.mark.parametrize('algorithm', available_algorithms())
def test_all_algorithms_with_maintenance(algorithm):
    class Model:
        def score(self, seqs):
            return np.array([[s.count('G'),s.count('C')] for s in seqs],dtype=float)

        def score_soft(self, p):
            return torch.stack([p[:,:,2].sum(1),p[:,:,1].sum(1)],1),None

    result = run_kernel(Model(),4,['AAAA','GGAA'],['maximize','minimize'],2,
                      algorithm=algorithm,stop_values=[(0,2),(0,4)],stop_tolerance=.5,
                      budget=128,diversity=0,generator_pretrain_epochs=1,
                      generator_update_epochs=1,den_seed_pretrain_epochs=1)
    obj = MultiTaskObjective(directions=['maximize','minimize'],stop_values=[(0,2),(0,4)])
    assert obj.feasible(result.activities).all()
    np.testing.assert_allclose(result.scores,obj.scalarize(result.activities))


@pytest.mark.parametrize('value', [0,9,True,1.5])
def test_round_patience_validation(value):
    with pytest.raises(ValueError,match='plateau_patience_rounds'):
        SaturationConfig(plateau_patience_rounds=value).resolve(1)


def test_stationary_archive_fastpath_with_bounds_matches_full_ranking():
    from sequence_optimization.api import CachedScorer
    from sequence_optimization.algorithms import _score, _rank_archive
    from sequence_optimization.advanced import _top_archive, _stored_utilities_are_stationary
    obj = MultiTaskObjective(directions=['maximize'],stop_values=[(0,2)])
    scorer = CachedScorer(ObjectiveScorer(
        CallableScorer(lambda seqs: [s.count('G') for s in seqs]),obj),budget=10)
    archive = {}
    _score(scorer,archive,np.array(['AAAA','GAAA','GGAA','GGGA','GGGG']))
    assert set(archive)=={'AAAA','GAAA','GGAA'}
    assert _stored_utilities_are_stationary(scorer)
    sequences,scores = _top_archive(archive,3,scorer)
    reference = _rank_archive(archive,scorer)
    assert list(sequences)==[s for s,_ in reference]
    np.testing.assert_array_equal(scores,[v[0] for _,v in reference])
