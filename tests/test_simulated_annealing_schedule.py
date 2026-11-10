import numpy as np
import pytest

from sequence_optimization import CallableScorer, OptimizationConfig, SaturationConfig
from sequence_optimization.algorithms import optimize


def test_default_schedule_runs_1000_iterations_with_sufficient_budget():
    result = optimize(
        CallableScorer(lambda ss: [s.count('G') for s in ss]),
        4, ['AAAA', 'CCCC'], ['maximize'], 2,
        algorithm='simulated_annealing', diversity=0, budget=2002,
    )
    assert result.metadata['iterations'] == 1000
    assert result.query_count == 2002
    assert result.trajectory[0]['temperature'] == pytest.approx(.1)
    assert result.trajectory[-1]['temperature'] == pytest.approx(.05)
    temperatures = np.array([row['temperature'] for row in result.trajectory])
    assert np.all(np.diff(temperatures) < 0)


def test_budget_truncates_iterations_and_temperature_uses_full_horizon():
    result = optimize(
        CallableScorer(lambda ss: np.zeros(len(ss))),
        2, ['AA', 'CC'], ['maximize'], 2,
        algorithm='simulated_annealing', diversity=0, budget=22,
    )
    assert result.metadata['iterations'] == 10
    assert result.trajectory[-1]['temperature'] == pytest.approx(.1 + 9/999*(.05-.1))


def test_saturation_stops_before_1000_iterations():
    result = optimize(
        CallableScorer(lambda ss: np.zeros(len(ss))),
        2, ['AA', 'CC'], ['maximize'], 2,
        algorithm='simulated_annealing', diversity=0,
        saturation=SaturationConfig(
            minimum_queries=2, plateau_patience_rounds=20,
            material_improvement=.01,
        ),
    )
    assert result.stop_reason == 'plateau'
    assert result.total_query_count == 42
    assert result.selected.metadata['iterations'] == 20
    assert result.selected.metadata['last_temperature'] == pytest.approx(
        .1 + 19/999*(.05-.1)
    )


@pytest.mark.parametrize('kwargs,match', [
    ({'annealing_iterations': 0}, 'annealing_iterations'),
    ({'annealing_iterations': True}, 'annealing_iterations'),
    ({'annealing_initial_temperature': 0}, 'annealing_initial_temperature'),
    ({'annealing_final_temperature': float('inf')}, 'annealing_final_temperature'),
    ({'annealing_initial_temperature': .05, 'annealing_final_temperature': .1},
     'must not exceed'),
])
def test_invalid_annealing_schedule(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OptimizationConfig(budget=10, **kwargs)


def test_uncapped_saturation_continues_after_cooling_horizon():
    result = optimize(
        CallableScorer(lambda ss: np.zeros(len(ss))),
        2, ['AA', 'CC'], ['maximize'], 2,
        algorithm='simulated_annealing', diversity=0, annealing_iterations=2,
        saturation=SaturationConfig(
            minimum_queries=2, plateau_patience_rounds=20,
            material_improvement=.01,
        ),
    )
    assert result.stop_reason == 'plateau'
    assert result.total_query_count == 42
    assert result.selected.metadata['iterations'] == 20
    assert result.selected.metadata['last_temperature'] == pytest.approx(.05)


def test_fixed_budget_keeps_explicit_iteration_limit():
    result = optimize(
        CallableScorer(lambda ss: np.zeros(len(ss))),
        2, ['AA', 'CC'], ['maximize'], 2,
        algorithm='simulated_annealing', diversity=0, annealing_iterations=2,
        budget=100,
    )
    assert result.query_count == 6
    assert result.metadata['iterations'] == 2
    assert [r['temperature'] for r in result.trajectory] == pytest.approx([.1, .05])
