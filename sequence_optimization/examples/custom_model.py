"""Wrap a channels-last model using A/G/C/T channel order."""

import torch

from sequence_optimization import (
    SequencePredictor,
    basesearch_optimize,
    random_seed_sequences,
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


class AGCTChannelsLastPredictor(SequencePredictor):
    def prepare_input(self, values):
        return values[:, :, [0, 2, 1, 3]]


def make_predictor(model):
    return AGCTChannelsLastPredictor(
        model,
        task_names=["activity"],
        device="cpu",
    )


def example():
    model = MyModel()
    # For a trained model, load its state_dict before wrapping it.
    predictor = make_predictor(model)
    return basesearch_optimize(
        predictor,
        length=20,
        initial=random_seed_sequences(20, 4, seed=7),
        objective=["maximize"],
        num_candidates=4,
        budget=40,
        seed=7,
    )
