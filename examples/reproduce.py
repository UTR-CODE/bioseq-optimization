"""Runtime settings for reproducing the four full notebook runs."""

from MotifSift._api import _objective_for
from sequence_optimization.saturation import SaturationConfig
from sequence_optimization.seeds import random_seed_sequences
from sequence_optimization.suite import generate_designs


def recorded_seeds(length):
    return random_seed_sequences(length, 100, seed=42)


def recorded_designs(case, model, length, initial, objective, **settings):
    settings_by_case = {
        "optimus": dict(max_batch_size=4096, max_soft_batch_size=128),
        "deepstarr": dict(saturation=SaturationConfig(
            fractions=(0.9,), minimum_queries=1000, plateau_patience_queries=4000,
        )),
        "mpralegnet": dict(saturation=SaturationConfig(
            fractions=(0.9,), minimum_queries=1000, plateau_patience_queries=4000,
        )),
        "saluki": dict(max_batch_size=256, max_soft_batch_size=16, batch_wait_ms=2),
    }
    return generate_designs(
        model, length, initial=initial, objective=_objective_for(model, objective),
        num_candidates=100, diversity=5, seed=42,
        **settings_by_case[case], **settings,
    )


def recorded_device(*, exact_arithmetic=False):
    import torch

    if exact_arithmetic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return "cuda" if torch.cuda.is_available() else "cpu"


def recorded_model(name):
    from bioseq_models import load_model

    optimus = name == "optimus5prime"
    return load_model(name, device=recorded_device(exact_arithmetic=optimus), batch_size=4096 if optimus else 128)
