"""The public workflow over the optimization and motif engines."""

from dataclasses import dataclass
from typing import Mapping, Sequence

from sequence_optimization.api import SequencePredictor as _SequencePredictor
from sequence_optimization.combined import CombinedScorer as _CombinedScorer
from sequence_optimization.framework import call_motifs as _call_motifs
from sequence_optimization.framework import select_candidates as _select_candidates
from sequence_optimization.objectives import CustomObjective
from sequence_optimization.objectives import MultiTaskObjective as _MultiTaskObjective
from sequence_optimization.suite import generate_designs as _generate_designs


class SequencePredictor(_SequencePredictor):
    """Adapt a PyTorch model that takes [batch, 4, length] A/C/G/T inputs.

    List the task names in the order returned by the model. For a different
    input layout, subclass this adapter and override ``prepare_input``.
    """

    def __init__(self, model, tasks):
        import torch

        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        tensor = next(model.parameters(), None)
        if tensor is None:
            tensor = next(model.buffers(), None)
        super().__init__(model, tasks=tasks, device=tensor.device if tensor is not None else "cpu")


class CombinedScorer(_CombinedScorer):
    """Present named single-output models as one multi-output scorer."""

    def __init__(self, models):
        super().__init__(models.values(), names=models.keys())


@dataclass(frozen=True)
class MultiTaskObjective:
    """Maximize selected outputs and minimize others, with optional named weights."""

    high: Sequence[str] = ()
    low: Sequence[str] = ()
    weights: Mapping[str, float] | None = None

    def _for_tasks(self, tasks):
        high = (self.high,) if isinstance(self.high, str) else tuple(self.high)
        low = (self.low,) if isinstance(self.low, str) else tuple(self.low)
        selected = high + low
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Specify at least one task, each in high or low only once")
        if not tasks or any(task not in tasks for task in selected):
            raise ValueError(f"Objective tasks {selected} must be among model tasks {tasks}")
        if self.weights is None:
            return _MultiTaskObjective(high=high, low=low)
        if not isinstance(self.weights, Mapping):
            raise TypeError("weights must map task names to non-negative numbers")
        weights = self.weights
        if weights.keys() - set(selected):
            raise ValueError("Specify weights only for tasks in high or low")
        ordered = [weights.get(task, 1) if task in selected else 0 for task in tasks]
        return _MultiTaskObjective(high=high, low=low, weights=ordered)


def _require_scorer(model):
    if not callable(getattr(model, "score", None)):
        raise TypeError("Use a bioseq_models scorer or wrap a PyTorch model with SequencePredictor(model, tasks)")


def _objective_for(model, objective):
    if isinstance(objective, str):
        if objective not in {"maximize", "minimize"}:
            raise ValueError("For one output, use objective='maximize' or objective='minimize'")
        return _MultiTaskObjective(directions=(objective,))
    if isinstance(objective, MultiTaskObjective):
        return objective._for_tasks(getattr(model, "task_names", None))
    if callable(objective):
        return CustomObjective(objective)
    return objective


def generate_designs(
    model, length, objective="maximize", *, initial=None, num_candidates=100,
    gain_fractions=None, score_threshold=None, algorithms=None, constraint=None,
    fixed_region=None, is_CDS_optimized=False, seed=0, output_dir=None,
):
    """Return one row per candidate, optimizer, and requested gain stage."""
    _require_scorer(model)
    return _generate_designs(
        model, length, initial=initial, objective=_objective_for(model, objective),
        num_candidates=num_candidates, gain_fractions=gain_fractions,
        score_threshold=score_threshold, algorithms=algorithms, constraint=constraint,
        fixed_region=fixed_region, is_CDS_optimized=is_CDS_optimized,
        seed=seed, output_dir=output_dir,
    )


def call_motifs(
    designs, model=None, *, reverse_complement=True, output_dir=None,
    objective=None, contributions=None, catalog=None, hits=None,
):
    """Discover motifs using the objective saved with the designs."""
    if model is not None:
        _require_scorer(model)
    return _call_motifs(
        designs, model=model, reverse_complement=reverse_complement,
        objective=_objective_for(model, objective), contributions=contributions,
        catalog=catalog, hits=hits, output_dir=output_dir,
    )


def select_candidates(designs, top_n=10, motifs=None, *, min_score=None, output_dir=None):
    """Return a panel of up to top_n candidates."""
    return _select_candidates(designs, top_n=top_n, motifs=motifs, min_score=min_score, output_dir=output_dir)
