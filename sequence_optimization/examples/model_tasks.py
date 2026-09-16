"""Python task recipes; model loading is confined to examples."""

from bioseq_models import load_model

from sequence_optimization import (
    CombinedScorer,
    genetic_algorithm_optimize,
    random_seed_sequences,
)

REFERENCE_CDS = "ATG" + "GCT" * 18 + "TAA"
FIXED_UTR5 = "ACGT" * 10
FIXED_UTR3 = "ACGT" * 10


def design_utr_code(region="utr5", *, budget=100, device="cpu"):
    fixed = {"utr5": FIXED_UTR5, "cds": REFERENCE_CDS, "utr3": FIXED_UTR3}
    if region not in fixed:
        raise ValueError("region must be utr5, cds, or utr3")
    fixed.pop(region)
    model = load_model("utr_code", design_region=region, device=device, **fixed)
    is_cds = region == "cds"
    initial = (
        [REFERENCE_CDS]
        if is_cds
        else random_seed_sequences(
            length={"utr5": 50, "utr3": 80}[region], count=4, seed=7
        )
    )
    return genetic_algorithm_optimize(
        model,
        length={"utr5": 50, "utr3": 80, "cds": 60}[region],
        initial=initial,
        objective=["maximize"],
        num_candidates=4,
        budget=budget,
        is_CDS_optimized=is_cds,
        seed=7,
    )


def design_cds_stability(*, joint=False, budget=100, device="cpu"):
    predictor = load_model("cds_stability", species="human", device=device)
    directions = ["maximize"]
    if joint:
        translation = load_model(
            "utr_code",
            design_region="cds",
            utr5=FIXED_UTR5,
            utr3=FIXED_UTR3,
            device=device,
        )
        predictor = CombinedScorer([predictor, translation], names=["stability", "rpf"])
        directions = ["maximize", "maximize"]
    return genetic_algorithm_optimize(
        predictor,
        length=len(REFERENCE_CDS),
        initial=[REFERENCE_CDS],
        objective=directions,
        num_candidates=4,
        budget=budget,
        is_CDS_optimized=True,
        seed=7,
    )


def design_deepstarr(*, budget=100, device="cpu"):
    return genetic_algorithm_optimize(
        load_model("deepstarr", device=device),
        length=249,
        initial=random_seed_sequences(249, 4, seed=7),
        objective=["minimize", "maximize"],
        num_candidates=4,
        budget=budget,
        seed=7,
    )


def design_dream(*, budget=100, device="cpu"):
    model = load_model("dream_legnet", variable_region_only=True, device=device)
    return genetic_algorithm_optimize(
        model,
        length=93,
        initial=random_seed_sequences(93, 4, seed=7),
        objective=["maximize"],
        num_candidates=4,
        budget=budget,
        seed=7,
    )


def design_cetnar(*, joint=False, budget=100, device="cpu"):
    predictor = load_model("cetnar_stability", device=device)
    directions = ["minimize"]
    if joint:
        translation = load_model(
            "utr_code",
            design_region="utr3",
            utr5=FIXED_UTR5,
            cds=REFERENCE_CDS,
            device=device,
        )
        predictor = CombinedScorer(
            [predictor, translation], names=["decay_rate", "rpf"]
        )
        directions = ["minimize", "maximize"]
    return genetic_algorithm_optimize(
        predictor,
        length=80,
        initial=random_seed_sequences(80, 4, seed=7),
        objective=directions,
        num_candidates=4,
        budget=budget,
        seed=7,
    )


def design_agarwal(target="k562", *, budget=100, device="cpu"):
    cells = ["k562", "hepg2", "wtc11"]
    if target not in cells:
        raise ValueError(f"target must be one of {cells}")
    predictor = CombinedScorer(
        [load_model(f"agarwal_{cell}", device=device) for cell in cells], names=cells
    )
    return genetic_algorithm_optimize(
        predictor,
        length=230,
        initial=random_seed_sequences(230, 4, seed=7),
        objective=["maximize" if cell == target else "minimize" for cell in cells],
        num_candidates=4,
        budget=budget,
        seed=7,
    )


def design_tissue(family="mouse", target="heart", *, budget=100, device="cpu"):
    tissues = {
        "mouse": ["heart", "limb", "midbrain", "forebrain", "hindbrain", "neural_tube"],
        "embryo": ["brain_vnc", "muscle", "epidermis", "gut"],
    }
    if family not in tissues or target not in tissues[family]:
        raise ValueError("choose a supported family and activity tissue")
    names = tissues[family]
    predictor = CombinedScorer(
        [
            load_model(f"deepstarr_{family}_{tissue}_activity", device=device)
            for tissue in names
        ],
        names=names,
    )
    return genetic_algorithm_optimize(
        predictor,
        length=1001,
        initial=random_seed_sequences(1001, 4, seed=7),
        objective=["maximize" if tissue == target else "minimize" for tissue in names],
        num_candidates=4,
        budget=budget,
        seed=7,
    )


def design_malinois(target="K562", *, budget=100, device="cpu"):
    cells = ["K562", "HepG2", "SKNSH"]
    if target not in cells:
        raise ValueError(f"target must be one of {cells}")
    return genetic_algorithm_optimize(
        load_model("malinois", device=device),
        length=200,
        initial=random_seed_sequences(200, 4, seed=7),
        objective=["maximize" if cell == target else "minimize" for cell in cells],
        num_candidates=4,
        budget=budget,
        seed=7,
    )
