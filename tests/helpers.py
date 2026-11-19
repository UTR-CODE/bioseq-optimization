"""Direct entry point for tests of a bounded optimizer kernel."""

from dataclasses import replace

from sequence_optimization.algorithms import _run_budget_segment
from sequence_optimization.api import OptimizationConfig
from sequence_optimization.objectives import MultiTaskObjective


def run_kernel(
    model, length, initial, objective, num_candidates, *, budget,
    algorithm="genetic_algorithm", alphabet="ACGT", seed=0, constraint=None,
    count_repeated=False, stop_values=None, stop_tolerance=0.5, **options,
):
    if stop_values is not None:
        if isinstance(objective, (list, tuple)):
            objective = MultiTaskObjective(directions=objective)
        if not isinstance(objective, MultiTaskObjective):
            raise ValueError("stop_values requires a directions objective")
        objective = replace(objective, stop_values=stop_values, stop_tolerance=stop_tolerance)
    config = OptimizationConfig(budget=budget, num_candidates=num_candidates, **options)
    return _run_budget_segment(
        algorithm, model, length, config, alphabet=alphabet, seed=seed,
        initial=initial, objective=objective, constraint=constraint,
        count_repeated=count_repeated,
    )
