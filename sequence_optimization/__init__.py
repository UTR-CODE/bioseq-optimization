"""Standalone, model-agnostic sequence optimization toolkit.

This package intentionally has no dependency on BioSeqBench datasets or
dataset-specific oracles. Wrap a model in a scorer, describe its sequence
space, choose seed sequences (or let the toolkit sample them), and run one or
more registered algorithms.
"""

from .api import (
    BudgetExceeded,
    CallableScorer,
    CachedScorer,
    OptimizationConfig,
    OptimizationResult,
    ScoreBatch,
    SequenceScorer,
    SequenceSpace,
    TorchSequenceScorer,
)
from .algorithms import (
    ISM_optimize,
    adalead_optimize,
    available_algorithms,
    genetic_algorithm_optimize,
    ism_optimize,
    optimize,
    random_optimize,
    register_algorithm,
    run,
    run_many,
    simulated_annealing_optimize,
)
from .objectives import TargetObjective
from .seeds import load_seed_sequences, random_seed_sequences

__all__ = [
    "BudgetExceeded", "CallableScorer", "CachedScorer", "OptimizationConfig",
    "OptimizationResult", "ScoreBatch", "SequenceScorer", "SequenceSpace",
    "TorchSequenceScorer", "available_algorithms", "register_algorithm", "run", "run_many",
    "TargetObjective", "optimize", "random_optimize", "ism_optimize", "ISM_optimize",
    "simulated_annealing_optimize", "genetic_algorithm_optimize", "adalead_optimize",
    "load_seed_sequences", "random_seed_sequences",
]
