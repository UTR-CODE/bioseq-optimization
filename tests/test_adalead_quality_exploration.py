import numpy as np
import pytest
from sequence_optimization import CallableScorer, OptimizationConfig, SaturationConfig
from sequence_optimization.algorithms import _quality_exploration_subset, optimize


def pick(sequences, scores, count, seed=42, **kwargs):
    return _quality_exploration_subset(np.asarray(sequences), np.asarray(scores), count,
                                      rng=np.random.default_rng(seed), **kwargs)


def test_best_anchor_and_unique_population_are_preserved():
    sequences=['AAAA','CCCC','GGGG','TTTT','AAAA']
    for seed in range(30):
        selected, scores=pick(sequences, [-2,-5,-4,-3,-2], 3, seed)
        assert selected[0] == 'AAAA' and scores[0] == -2
        assert len(selected) == len(set(selected)) == 3
    assert len(pick(sequences, [-2,-5,-4,-3,-2], 100)[0]) == 4


def test_relative_ranks_are_affine_and_order_invariant():
    sequences=np.array(['AAAA','CCCC','GGGG','TTTT','ACGT','TGCA'])
    scores=np.array([-7.,-4.,-3.,-2.,-1.,0.])
    original=pick(sequences,scores,4)
    transformed=pick(sequences[::-1],(3*scores+100)[::-1],4)
    np.testing.assert_array_equal(original[0], transformed[0])
    np.testing.assert_allclose(original[1]*3+100, transformed[1])


def test_quality_is_probabilistic_with_preference_for_higher_rank():
    counts=dict(CCCC=0,GGGG=0,TTTT=0)
    for seed in range(400):
        chosen,_=pick(['AAAA','CCCC','GGGG','TTTT'],[4,3,2,1],2,seed,exploration_fraction=0)
        counts[chosen[1]]+=1
    assert counts['CCCC'] > counts['GGGG'] > counts['TTTT'] > 0


def test_exploration_reaches_low_quality_distant_candidates():
    far=0
    for seed in range(200):
        chosen,_=pick(['AAAAAAAA','AAAAAAAC','GGGGGGGG'],[100,99,-100],2,seed,
                       exploration_fraction=.2,quality_pool_multiplier=1)
        far+=chosen[-1]=='GGGGGGGG'
    assert far > 180


def test_high_quality_power_does_not_eliminate_numeric_sampling_support():
    selected,_=pick(['AAAA','CCCC','GGGG','TTTT'],[4,3,2,1],3,
                    exploration_fraction=0,quality_power=10000)
    assert len(set(selected)) == 3


@pytest.mark.parametrize('kwargs', [dict(adalead_exploration_fraction=-.1),
    dict(adalead_exploration_fraction=1),dict(adalead_exploration_fraction=True),
    dict(adalead_quality_power=0),dict(adalead_quality_power=float('nan'))])
def test_invalid_mixture_settings(kwargs):
    with pytest.raises(ValueError,match='adalead_'):
        OptimizationConfig(budget=10,**kwargs)


def test_plateau_metadata_and_candidates_use_quality_exploration():
    result=optimize(CallableScorer(lambda ss:[s.count('G') for s in ss]),8,
        ['AAAAAAAA','CCCCCCCC','TTTTTTTT','ACACACAC'],['maximize'],4,
        algorithm='adalead',seed=42,diversity=1,
        saturation=SaturationConfig(minimum_queries=4))
    assert result.stop_reason=='plateau'
    for panel in [*result.selections.values(),result.knee,result.knee_fraction_result]:
        assert panel.metadata['parent_selection']=='rank_weighted_quality_with_novelty_exploration'
        assert panel.metadata['survivor_selection']==panel.metadata['parent_selection']
        assert panel.metadata['exploration_fraction']==.2
        assert panel.metadata['best_anchor_preserved'] is True
        assert 'parent_cluster_cap' not in panel.metadata
        assert len(panel.sequences)==4
