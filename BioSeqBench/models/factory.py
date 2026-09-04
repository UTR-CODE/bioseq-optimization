from __future__ import annotations

from torch import nn

from .deepstarr import DeepSTARRPaper
from .agarwal2025_lentimpra import AgarwalMPRACNN, MPRALegNet
from .cetnar2024_mrna_stability import CetnarStabilityCNNBiLSTM
from .ecoli_promoter import EcoliPromoterCNN
from .zhou2026_l3_promoters import L3ConditionalCNN, L3PositionAwareCNN
from .gb1 import GB1FlipCNN
from .gfp import GFPDesignBenchTransformer
from .igvf_ccre_mpra import IGVFLegNet
from .kosuri2013_composability import KosuriComposabilityCNN
from .openvaccine import OpenVaccineGCNGRU
from .protein import ProteinCNN, ProteinTransformer
from .yeast_gpra import YeastGPRACNN
from .yeast_gpra_dream2022 import LegNetAutosome
from .parm import PARMCNN, PARMExactCNN
from .tfbind8 import TFBind8DeepBind, TFBind8DyNACNN
from .utr5_mpra import Optimus5PrimePaper
from .weakseqcnn import WeakSeqCNN


def build_model(name: str, *, output_dim: int = 1, num_cell_types: int = 1, max_length: int = 1024) -> nn.Module:
    key = name.lower().replace("-", "_")
    constructors = {
        "deepbind": lambda: TFBind8DeepBind(output_dim=output_dim),
        "tfbind8_deepbind": lambda: TFBind8DeepBind(output_dim=output_dim),
        "dynappo_cnn": lambda: TFBind8DyNACNN(output_dim=output_dim),
        "tfbind8_dynappo_cnn": lambda: TFBind8DyNACNN(output_dim=output_dim),
        "optimus5prime": lambda: Optimus5PrimePaper(output_dim=output_dim),
        "weakseqcnn": lambda: WeakSeqCNN(),
        "weakseqcnn_v1": lambda: WeakSeqCNN(),
        "deepstarr": lambda: DeepSTARRPaper(output_dim=output_dim),
        "gb1_flip_cnn": lambda: GB1FlipCNN(output_dim=output_dim),
        "gfp_transformer": lambda: GFPDesignBenchTransformer(output_dim=output_dim, max_length=max_length),
        "protein_cnn": lambda: ProteinCNN(output_dim=output_dim),
        "protein_transformer": lambda: ProteinTransformer(output_dim=output_dim, max_length=max_length),
        "openvaccine": lambda: OpenVaccineGCNGRU(output_dim=output_dim),
        "openvaccine_gcn_gru": lambda: OpenVaccineGCNGRU(output_dim=output_dim),
        "conditional_cnn": lambda: IGVFLegNet(num_cell_types=num_cell_types, output_dim=output_dim),
        "igvf_legnet": lambda: IGVFLegNet(num_cell_types=num_cell_types, output_dim=output_dim),
        "cetnar_cnn_bilstm": lambda: CetnarStabilityCNNBiLSTM(output_dim=output_dim),
        "cetnar2024_cnn_bilstm": lambda: CetnarStabilityCNNBiLSTM(output_dim=output_dim),
        "kosuri_composability_cnn": lambda: KosuriComposabilityCNN(output_dim=output_dim),
        "ecoli_promoter_cnn": lambda: EcoliPromoterCNN(output_dim=output_dim),
        "zhou_l3_cnn": lambda: L3PositionAwareCNN(output_dim=output_dim, length=max_length),
        "zhou_l3_conditional_cnn": lambda: L3ConditionalCNN(
            num_sublibraries=num_cell_types, output_dim=output_dim, length=max_length,
        ),
        "yeast_gpra_cnn": lambda: YeastGPRACNN(output_dim=output_dim),
        "autosome_legnet": lambda: LegNetAutosome(),
        "parm_cnn": lambda: PARMCNN(output_dim=output_dim),
        "parm_exact": lambda: PARMExactCNN(),
        "agarwal_mpra_cnn": lambda: AgarwalMPRACNN(output_dim=output_dim),
        "mpralegnet": lambda: MPRALegNet(),
        "agarwal_mpralegnet": lambda: MPRALegNet(),
    }
    if key not in constructors:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(constructors)}")
    return constructors[key]()
