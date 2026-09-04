from .cetnar2024 import Cetnar2024MRNAStabilityLoader
from .agarwal2025 import (
    Agarwal2025LentiMPRALoader,
    Agarwal2025LentiMPRAHepG2Loader,
    Agarwal2025LentiMPRAK562Loader,
    Agarwal2025LentiMPRAWTC11Loader,
)
from .deepstarr import DeepSTARRLoader
from .gb1 import GB1Loader
from .gfp import GFPLoader
from .igvf import IGVFLoader
from .kosuri2013 import Kosuri2013ComposabilityLoader
from .openvaccine import OpenVaccineLoader
from .tfbind8 import TFBind8Loader
from .tfbind10 import TFBind10Loader
from .utr import UTRLoader
from .zhao2022 import Zhao2022PtrcLoader
from .zhou2026 import Zhou2026PromoterLoader

LOADERS = {
    "agarwal2025_lentimpra": Agarwal2025LentiMPRALoader,
    "agarwal2025_lentimpra_hepg2": Agarwal2025LentiMPRAHepG2Loader,
    "agarwal2025_lentimpra_k562": Agarwal2025LentiMPRAK562Loader,
    "agarwal2025_lentimpra_wtc11": Agarwal2025LentiMPRAWTC11Loader,
    "tfbind8": TFBind8Loader,
    "tfbind10": TFBind10Loader,
    "gb1": GB1Loader,
    "gfp": GFPLoader,
    "utr5_mpra": UTRLoader,
    "igvf_ccre_mpra": IGVFLoader,
    "deepstarr": DeepSTARRLoader,
    "openvaccine": OpenVaccineLoader,
    "cetnar2024_mrna_stability": Cetnar2024MRNAStabilityLoader,
    "kosuri2013_composability": Kosuri2013ComposabilityLoader,
    "zhao2022_ptrc_promoters": Zhao2022PtrcLoader,
    "zhou2026_l3_promoters": Zhou2026PromoterLoader,
}

__all__ = ["LOADERS"]
