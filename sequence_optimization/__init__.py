"""Standalone, model-agnostic sequence optimization toolkit.

Supply a PyTorch sequence-to-activity model, starting sequences, an objective,
and a candidate count. Input and output transforms connect the optimizer to
the model used in each experiment.
"""

from .api import (
    BudgetExceeded,
    CallableScorer,
    CachedScorer,
    SequencePredictor,
    OptimizationConfig,
    OptimizationResult,
    ScoreBatch,
    SequenceScorer,
    SequenceSpace,
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
from .constraints import ConstraintFunction, ConstraintSpec
from .onehot import onehot
from .objectives import MultiTaskObjective, TargetObjective
from .saturation import SaturationConfig, SaturationResult
from .seeds import load_seed_sequences, random_seed_sequences
from .suite import DIFFERENTIABLE_ALGORITHMS, generate_designs
from .families import FamilySelectionResult, kmer_position_features, select_diverse_families
from .exploration import (
    DiversitySearchConfig,
    DiversitySearchResult,
    MotifPositionDescriptor,
    generate_diverse_designs,
)
from .panel import ClusteredDesignPanel, cluster_designs
from .attribution import ISMContributions, ism_contributions

__all__ = [
    "ISMContributions",
    "ism_contributions",
    "ClusteredDesignPanel",
    "cluster_designs",
    "DiversitySearchConfig",
    "DiversitySearchResult",
    "MotifPositionDescriptor",
    "generate_diverse_designs",
    "BudgetExceeded",
    "CallableScorer",
    "CombinedScorer",
    "ConstraintFunction",
    "ConstraintSpec",
    "onehot",
    "CachedScorer",
    "OptimizationConfig",
    "OptimizationResult",
    "ScoreBatch",
    "SequenceScorer",
    "SequenceSpace",
    "SequencePredictor",
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
    "vanilla_activation_maximization_optimize",
    "fast_seqprop_optimize",
    "den_optimize",
    "SaturationConfig",
    "SaturationResult",
    "load_seed_sequences",
    "random_seed_sequences",
    "DIFFERENTIABLE_ALGORITHMS",
    "generate_designs",
    "FamilySelectionResult",
    "select_diverse_families",
    "kmer_position_features",
]
