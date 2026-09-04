"""Model-guided biological-sequence optimization algorithms."""

from .core import (
    BudgetExceeded,
    CachedScorer,
    CallableScorer,
    TorchSequenceScorer,
    OptimizationConfig,
    OptimizationResult,
    ScoreBatch,
    SequenceSpace,
    available_optimizers,
    register_optimizer,
    run_optimizer,
    run_optimizers,
)

# Backward-compatible dataset benchmark entry points.
from .tfbind8_benchmark import (
    TFBind8OptimizationConfig,
    run_tfbind8_benchmark,
)

ALGORITHM_NAMES = available_optimizers()

__all__ = [
    "ALGORITHM_NAMES", "BudgetExceeded", "CachedScorer", "CallableScorer", "TorchSequenceScorer",
    "OptimizationConfig", "OptimizationResult", "ScoreBatch", "SequenceSpace",
    "available_optimizers", "register_optimizer", "run_optimizer", "run_optimizers",
    "TFBind8OptimizationConfig", "run_tfbind8_benchmark",
]
