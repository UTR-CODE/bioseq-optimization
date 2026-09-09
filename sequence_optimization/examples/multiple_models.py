"""Python examples: one model high and two models low."""

from bioseq_models import load_model
from sequence_optimization import CombinedScorer, basesearch_optimize


def lentimpra_example():
    scorer = CombinedScorer(
        [
            load_model(name, device="cpu")
            for name in ["agarwal_k562", "agarwal_hepg2", "agarwal_wtc11"]
        ],
        names=["K562", "HepG2", "WTC11"],
    )
    return basesearch_optimize(
        scorer,
        length=230,
        objective=["maximize", "minimize", "minimize"],
        num_candidates=4,
        budget=100,
        seed=7,
    )


def mouse_example():
    scorer = CombinedScorer(
        [
            load_model(name, device="cpu")
            for name in [
                "deepstarr_mouse_heart_activity",
                "deepstarr_mouse_limb_activity",
                "deepstarr_mouse_midbrain_activity",
            ]
        ],
        names=["heart", "limb", "midbrain"],
    )
    return basesearch_optimize(
        scorer,
        length=1001,
        objective=["maximize", "minimize", "minimize"],
        num_candidates=4,
        budget=100,
        seed=7,
    )
