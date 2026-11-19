from tests.helpers import run_kernel

import numpy as np
import pytest

from sequence_optimization import CallableScorer, OptimizationConfig
from sequence_optimization.algorithms import _mutate_expected
from sequence_optimization.api import SequenceSpace
from sequence_optimization.objectives import ObjectiveScorer
from sequence_optimization.saturation import _SaturationMonitor, _TrackingScorer


def test_rollout_accepts_neutral_child_and_accumulates_mutations(monkeypatch):
    import sequence_optimization.algorithms as module
    children = iter(['GAAA', 'GGAA', 'GGGA'])
    monkeypatch.setattr(module, '_mutate_expected',
                        lambda sequences, space, expected, rng: np.array([next(children)]))
    scores = {'AAAA': 0., 'CCCC': -1., 'GAAA': 0., 'GGAA': 1., 'GGGA': -.5}
    result = run_kernel(CallableScorer(lambda ss: [scores[s] for s in ss]), 4,
                      ['AAAA','CCCC'], ['maximize'], 2, algorithm='adalead',
                      budget=5, diversity=0, elite_fraction=.5,
                      adalead_max_rollout=3, adalead_lineages=1)
    assert 'GGAA' in result.sequences
    assert result.metadata['accepted_children'] == 2
    assert result.metadata['rollout_allow_equal'] is True


def test_rollout_compares_to_fixed_root_not_previous_child(monkeypatch):
    import sequence_optimization.algorithms as module
    children = iter(['GAAA', 'GGAA'])
    monkeypatch.setattr(module, '_mutate_expected',
                        lambda sequences, space, expected, rng: np.array([next(children)]))
    scores = {'AAAA': 0., 'CCCC': -1., 'GAAA': 2., 'GGAA': 1.}
    result = run_kernel(CallableScorer(lambda ss: [scores[s] for s in ss]), 4,
                      ['AAAA','CCCC'], ['maximize'], 2, algorithm='adalead',
                      budget=4, diversity=0, elite_fraction=.5,
                      adalead_max_rollout=2, adalead_lineages=1)
    assert result.metadata['accepted_children'] == 2


def test_rollout_depth_batches_count_as_one_patience_round():
    objective = ObjectiveScorer(CallableScorer(lambda ss: [s.count('G') for s in ss]), ['maximize'])
    monitor = _SaturationMonitor(1, 1000, 1000, .01, 0, patience_rounds=20)
    scorer = _TrackingScorer(objective, 100, monitor=monitor)
    scorer.score(['AAAA'])
    assert monitor.observation_round == 1
    scorer.begin_observation_group()
    scorer.score(['GAAA'])
    scorer.score(['GGAA'])
    scorer.score(['GGGA'])
    assert monitor.observation_round == 1
    scorer.end_observation_group()
    assert monitor.observation_round == 2


def test_rollout_respects_fixed_positions():
    result = run_kernel(CallableScorer(lambda ss: [s.count('G') for s in ss]), 8,
                      ['AAAATTTT','AAAAGGGG'], ['maximize'], 2, algorithm='adalead',
                      budget=80, diversity=0, fixed_region=['0:4'],
                      adalead_mu=3, adalead_max_rollout=4)
    assert all(s[:4] == 'AAAA' for s in result.sequences)


def test_only_feasible_accepted_children_continue_rollout(monkeypatch):
    import sequence_optimization.algorithms as module
    original = module._mutate_expected
    observed = []

    def checked(sequences, space, expected, rng):
        observed.extend(sequences)
        assert all(s.count('G') <= 2 for s in sequences)
        return original(sequences, space, expected, rng)

    monkeypatch.setattr(module, '_mutate_expected', checked)
    result = run_kernel(CallableScorer(lambda ss: [s.count('G') for s in ss]), 4,
                      ['GGAA','GGGG'], ['maximize'], 2, algorithm='adalead',
                      stop_values=[2], budget=128, diversity=0)
    assert observed
    assert (result.activities <= 2).all()


def test_mutation_expected_can_make_multiple_changes_and_preserves_space(monkeypatch):
    space = SequenceSpace(8)
    rng = np.random.default_rng(3)
    values = _mutate_expected(np.array(['AAAAAAAA'] * 200), space, 4, rng)
    distances = np.array([sum(a != b for a,b in zip('AAAAAAAA',s)) for s in values])
    assert distances.max() > 1
    space.validate(values)


@pytest.mark.parametrize('kwargs, match', [
    ({'adalead_mu': 0}, 'adalead_mu'),
    ({'adalead_mu': True}, 'adalead_mu'),
    ({'adalead_max_rollout': 0}, 'adalead_max_rollout'),
    ({'adalead_allow_equal': 1}, 'adalead_allow_equal'),
    ({'adalead_lineages': 0}, 'adalead_lineages'),
])
def test_adalead_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OptimizationConfig(budget=10, **kwargs)
