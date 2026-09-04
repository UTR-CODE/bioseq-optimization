"""Catalog of model implementations collected elsewhere in this project.

The catalog is metadata only: importing ``sequence_optimization`` never imports
BioSeqBench datasets or model code. Examples can be resolved lazily when a user
chooses one of these model shells.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class ModelExample:
    name: str
    import_path: str
    class_name: str
    alphabet: str
    length: int | None
    description: str

    def load(self) -> Any:
        module_name, separator, class_name = self.import_path.partition(":")
        if not separator:
            raise ValueError(f"invalid import path: {self.import_path}")
        return getattr(import_module(module_name), class_name)


MODEL_EXAMPLES = {
    "weakseqcnn_v1": ModelExample(
        "weakseqcnn_v1", "BioSeqBench.models.weakseqcnn:WeakSeqCNN", "WeakSeqCNN", "ACGT", None,
        "standardized 194,882-parameter surrogate CNN",
    ),
    "optimus5prime": ModelExample(
        "optimus5prime", "BioSeqBench.models.utr5_mpra:Optimus5PrimePaper", "Optimus5PrimePaper", "ACGT", 50,
        "human 5′UTR Optimus strong model",
    ),
    "tfbind8": ModelExample(
        "tfbind8", "BioSeqBench.models.tfbind8:TFBind8DeepBind", "TFBind8DeepBind", "ACGT", 8,
        "TFBind8 DeepBind-style CNN",
    ),
    "dream_legnet": ModelExample(
        "dream_legnet", "BioSeqBench.models.yeast_gpra_dream2022:LegNetAutosome", "LegNetAutosome", "ACGT", 110,
        "DREAM promoter Autosome/LegNet model",
    ),
    "deepstarr": ModelExample(
        "deepstarr", "BioSeqBench.models.deepstarr:DeepSTARRPaper", "DeepSTARRPaper", "ACGT", 249,
        "DeepSTARR enhancer activity model",
    ),
    "parm": ModelExample(
        "parm", "BioSeqBench.models.parm:PARMCNN", "PARMCNN", "ACGT", None,
        "PARM regulatory-DNA activity CNN",
    ),
    "agarwal_mpralegnet": ModelExample(
        "agarwal_mpralegnet", "BioSeqBench.models.agarwal2025_lentimpra:MPRALegNet", "MPRALegNet", "ACGT", 230,
        "Agarwal lentiMPRA LegNet model",
    ),
    "kosuri_composability": ModelExample(
        "kosuri_composability", "BioSeqBench.models.kosuri2013_composability:KosuriComposabilityCNN", "KosuriComposabilityCNN", "ACGT", None,
        "Kosuri promoter/RBS composability CNN",
    ),
    "zhao_ptrc": ModelExample(
        "zhao_ptrc", "BioSeqBench.models.zhao2022_ptrc_promoters:PtrcPromoterCNN", "PtrcPromoterCNN", "ACGT", None,
        "Zhao Ptrc promoter model",
    ),
    "zhou_l3": ModelExample(
        "zhou_l3", "BioSeqBench.models.zhou2026_l3_promoters:L3PositionAwareCNN", "L3PositionAwareCNN", "ACGT", None,
        "Zhou L3 promoter model",
    ),
}


def available_model_examples() -> tuple[str, ...]:
    return tuple(sorted(MODEL_EXAMPLES))
