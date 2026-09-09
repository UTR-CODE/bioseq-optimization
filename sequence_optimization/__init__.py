"""Standalone, model-agnostic sequence optimization toolkit.

Wrap a model in a scorer, supply the sequence length, choose seed sequences
(or let the toolkit sample them), and run one or more registered algorithms.
Model architectures and weights live in the separate ``bioseq-models``
package.
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
    as_score_batch,
    normalize_nucleotide_sequences,
    pad_nucleotide_sequences,
)
from .algorithms import (
    ISM_optimize,
    adalead_optimize,
    available_algorithms,
    basesearch_optimize,
    beam_search_optimize,
    cbas_optimize,
    cross_entropy_optimize,
    den_optimize,
    fast_seqprop_optimize,
    genetic_algorithm_optimize,
    ism_optimize,
    ledidi_optimize,
    optimize,
    random_optimize,
    register_algorithm,
    run,
    run_many,
    simulated_annealing_optimize,
    vanilla_activation_maximization_optimize,
)
from .combined import CombinedScorer
from .onehot import onehot
from .objectives import MultiTaskObjective, TargetObjective
from .saturation import SaturationConfig, SaturationResult, optimize_to_saturation
from .seeds import load_seed_sequences, random_seed_sequences

__all__ = [
    "BudgetExceeded",
    "CallableScorer",
    "CombinedScorer",
    "onehot",
    "CachedScorer",
    "OptimizationConfig",
    "OptimizationResult",
    "ScoreBatch",
    "SequenceScorer",
    "SequenceSpace",
    "TorchSequenceScorer",
    "DifferentiableTorchSequenceScorer",
    "as_score_batch",
    "available_algorithms",
    "basesearch_optimize",
    "normalize_nucleotide_sequences",
    "pad_nucleotide_sequences",
    "register_algorithm",
    "run",
    "run_many",
    "TargetObjective",
    "MultiTaskObjective",
    "optimize",
    "random_optimize",
    "ism_optimize",
    "ledidi_optimize",
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
