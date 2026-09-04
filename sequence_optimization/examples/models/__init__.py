"""Adapters/examples for models collected by the BioSeqBench project.

The core optimizer does not import this module. These examples show how a
checkpoint-backed model can be exposed through the generic scorer shell.
"""

from .dream_legnet import DreamLegNetExample
from .deepstarr import DEFAULT_DEEPSTARR_CHECKPOINT, DeepSTARRExample
from .tfbind8 import TFBind8Example
from .utr5 import DEFAULT_OPTIMUS_CHECKPOINT, Optimus5PrimeExample, WeakSeqCNNExample
from .catalog import MODEL_EXAMPLES, ModelExample, available_model_examples

__all__ = [
    "DeepSTARRExample", "DEFAULT_DEEPSTARR_CHECKPOINT", "DreamLegNetExample", "TFBind8Example",
    "Optimus5PrimeExample", "WeakSeqCNNExample", "DEFAULT_OPTIMUS_CHECKPOINT",
    "MODEL_EXAMPLES", "ModelExample", "available_model_examples",
]
