"""Genetic optimization of a three-output CNN."""

import torch
from torch import nn

from sequence_optimization import (
    SequencePredictor,
    genetic_algorithm_optimize,
    random_seed_sequences,
)


class ThreeOutputCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 3, kernel_size=1)
        # Demonstration outputs: G, C and T fractions, each centered at 0.25.
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 2, 0] = 1
            self.conv.weight[1, 1, 0] = 1
            self.conv.weight[2, 3, 0] = 1
            self.conv.bias.fill_(-0.25)

    def forward(self, values):
        return self.conv(values).mean(dim=2)


def example():
    predictor = SequencePredictor(
        ThreeOutputCNN(), task_names=["output_1", "output_2", "output_3"], device="cpu"
    )
    return genetic_algorithm_optimize(
        predictor,
        length=20,
        initial=random_seed_sequences(20, 4, seed=7),
        objective=["maximize", "minimize", "ignore"],
        num_candidates=4,
        budget=80,
        mutation_rate=0.1,
        seed=7,
    )
