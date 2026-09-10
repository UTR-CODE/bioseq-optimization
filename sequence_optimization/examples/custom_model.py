"""Wrap a channels-last model using A/G/C/T channel order."""

from functools import partial

import torch

from sequence_optimization import (
    DifferentiableSequencePredictor,
    SequencePredictor,
    basesearch_optimize,
    onehot,
)


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.readout.weight.copy_(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))

    def forward(self, values):
        # Input: [N, L, 4], channels A/G/C/T. Example weights reward G.
        return self.readout(values.mean(dim=1))


def make_predictor(model, *, differentiable=False):
    cls = DifferentiableSequencePredictor if differentiable else SequencePredictor
    options = {}
    if differentiable:
        # Optimizers supply [N, L, A/C/G/T]; adapt without detaching gradients.
        options["soft_input_transform"] = lambda values: values[:, :, [0, 2, 1, 3]]
    return cls(
        model,
        onehot=partial(onehot, alphabet="AGCT", channels_first=False),
        task_names=["activity"],
        device="cpu",
        **options,
    )


def example():
    model = MyModel()
    # For a trained model, load its state_dict before wrapping it.
    predictor = make_predictor(model)
    return basesearch_optimize(
        predictor,
        length=20,
        objective=["maximize"],
        num_candidates=4,
        budget=40,
        seed=7,
    )
