"""Three custom CNN configurations with a shared genetic optimizer."""

import torch
from torch import nn

from sequence_optimization import (
    CombinedScorer,
    SequencePredictor,
    genetic_algorithm_optimize,
    random_seed_sequences,
)
from .multitask_cnn import ThreeOutputCNN


class SingleOutputCNN(nn.Module):
    def __init__(self, base="G"):
        super().__init__()
        self.conv = nn.Conv1d(4, 1, 1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, "ACGT".index(base), 0] = 1
            self.conv.bias.fill_(-0.25)

    def forward(self, values):
        return self.conv(values).mean(dim=2)


def single_task():
    return genetic_algorithm_optimize(
        SequencePredictor(SingleOutputCNN()),
        length=20,
        initial=random_seed_sequences(20, 4, seed=7),
        objective=["maximize"],
        num_candidates=4,
        budget=80,
        seed=7,
    )


def multi_task():
    return genetic_algorithm_optimize(
        SequencePredictor(ThreeOutputCNN()),
        length=20,
        initial=random_seed_sequences(20, 4, seed=7),
        objective=["maximize", "minimize", "ignore"],
        num_candidates=4,
        budget=80,
        seed=7,
    )


def independent_models():
    predictor = CombinedScorer(
        [SequencePredictor(SingleOutputCNN(base)) for base in "GCT"],
        names=["model_a", "model_b", "model_c"],
    )
    return genetic_algorithm_optimize(
        predictor,
        length=20,
        initial=random_seed_sequences(20, 4, seed=7),
        objective=["maximize", "minimize", "minimize"],
        num_candidates=4,
        budget=80,
        seed=7,
    )
