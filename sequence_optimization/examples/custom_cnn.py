"""Optimize a minimal one-output CNN without any external model package."""

import torch
from torch import nn

from sequence_optimization import (
    SequencePredictor,
    genetic_algorithm_optimize,
)


class MinimalCNN(nn.Module):
    """A one-layer CNN whose example weights reward G nucleotides."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(4, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.convolution.weight.zero_()
            self.convolution.weight[0, 2, 0] = 1.0

    def forward(self, inputs):
        return self.convolution(inputs).mean(dim=2)


def main() -> None:
    # In a real application, construct the trained architecture and load its
    # checkpoint before wrapping it in SequencePredictor.
    scorer = SequencePredictor(
        MinimalCNN(),
        device="cpu",
        task_names=("activity",),
    )
    result = genetic_algorithm_optimize(
        scorer,
        length=20,
        num_candidates=4,
        budget=40,
        mutation_rate=0.1,
        seed=7,
    )
    print(result.to_frame().to_string(index=False))


if __name__ == "__main__":
    main()
