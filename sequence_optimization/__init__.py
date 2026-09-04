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
    DifferentiableTorchSequenceScorer,
    OptimizationConfig,
    OptimizationResult,
    ScoreBatch,
    SequenceScorer,
    SequenceSpace,
    TorchSequenceScorer,
    normalize_nucleotide_sequences,
)
from .algorithms import (
    ISM_optimize,
    adalead_optimize,
    available_algorithms,
    beam_search_optimize,
    cbas_optimize,
    cross_entropy_optimize,
    den_optimize,
    fast_seqprop_optimize,
    genetic_algorithm_optimize,
    ism_optimize,
    optimize,
    random_optimize,
    register_algorithm,
    run,
    run_many,
    simulated_annealing_optimize,
    vanilla_activation_maximization_optimize,
)
from .objectives import TargetObjective
from .pretrained import (
    PretrainedModelInfo,
    available_models,
    load_model,
    model_info,
)
from .saturation import SaturationConfig, SaturationResult, optimize_to_saturation
from .seeds import load_seed_sequences, random_seed_sequences

__all__ = [
    "BudgetExceeded",
    "CallableScorer",
    "CachedScorer",
    "OptimizationConfig",
    "OptimizationResult",
    "ScoreBatch",
    "SequenceScorer",
    "SequenceSpace",
    "TorchSequenceScorer",
    "DifferentiableTorchSequenceScorer",
    "available_algorithms",
    "normalize_nucleotide_sequences",
    "register_algorithm",
    "run",
    "run_many",
    "TargetObjective",
    "PretrainedModelInfo",
    "available_models",
    "model_info",
    "load_model",
    "optimize",
    "random_optimize",
    "ism_optimize",
    "ISM_optimize",
    "simulated_annealing_optimize",
    "genetic_algorithm_optimize",
    "adalead_optimize",
    "beam_search_optimize",
    "cross_entropy_optimize",
    "cbas_optimize",
    "vanilla_activation_maximization_optimize",
    "fast_seqprop_optimize",
    "den_optimize",
    "SaturationConfig",
    "SaturationResult",
    "optimize_to_saturation",
    "load_seed_sequences",
    "random_seed_sequences",
]
