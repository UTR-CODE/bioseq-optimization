"""Adapters/examples for models collected by the BioSeqBench project.

The core optimizer does not import this module. These examples show how a
checkpoint-backed model can be exposed through the generic scorer shell.
"""

from .dream_legnet import DEFAULT_DREAM_LEGNET_CHECKPOINT, DreamLegNetExample
from .deepstarr import DEFAULT_DEEPSTARR_CHECKPOINT, DeepSTARRExample
from .tfbind8 import TFBind8Example
from .utr5 import DEFAULT_OPTIMUS_CHECKPOINT, Optimus5PrimeExample, WeakSeqCNNExample
from .zhao_ptrc import DEFAULT_ZHAO_PTRC_CHECKPOINT, ZhaoPtrcExample
from .zhou_l3 import DEFAULT_ZHOU_L3_CHECKPOINT, ZhouL3Example
from .catalog import MODEL_EXAMPLES, ModelExample, available_model_examples

__all__ = [
    "DeepSTARRExample",
    "DEFAULT_DEEPSTARR_CHECKPOINT",
    "DreamLegNetExample",
    "DEFAULT_DREAM_LEGNET_CHECKPOINT",
    "TFBind8Example",
    "Optimus5PrimeExample",
    "WeakSeqCNNExample",
    "DEFAULT_OPTIMUS_CHECKPOINT",
    "ZhaoPtrcExample",
    "DEFAULT_ZHAO_PTRC_CHECKPOINT",
    "ZhouL3Example",
    "DEFAULT_ZHOU_L3_CHECKPOINT",
    "MODEL_EXAMPLES",
    "ModelExample",
    "available_model_examples",
]
