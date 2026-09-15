import numpy as np
import torch

from sequence_optimization import (
    SequencePredictor,
    onehot,
)
from sequence_optimization.examples.custom_model import MyModel, example, make_predictor


def test_custom_numpy_onehot_layout_and_model_identity():
    model = MyModel()
    weights = model.readout.weight.detach().clone()
    predictor = make_predictor(model)
    assert predictor.model is model
    np.testing.assert_allclose(predictor.score(["AGGT", "AAAA"]).scores, [[0.5], [0]])
    torch.testing.assert_close(model.readout.weight, weights)
    assert not model.training
    assert example().activities.shape == (4, 1)


def test_custom_onehot_matches_differentiable_predictions():
    predictor = make_predictor(MyModel())
    inputs = torch.from_numpy(onehot(["AGGT", "AAAA"])).requires_grad_()
    scores, _ = predictor.score_soft(inputs)
    np.testing.assert_allclose(
        scores.detach(), predictor.score(["AGGT", "AAAA"]).scores
    )
    scores.sum().backward()
    np.testing.assert_allclose(inputs.grad[:, :, 2], 0.25)
    assert inputs.grad[:, :, [0, 1, 3]].eq(0).all()


def test_sequence_predictor_exposes_both_scoring_paths():
    predictor = SequencePredictor(MyModel())
    assert callable(predictor.score)
    assert callable(predictor.score_soft)


def test_subclass_prepares_extra_channels_for_both_paths():
    class FiveChannelModel(torch.nn.Module):
        def forward(self, values):
            assert values.shape[1] == 5
            return values[:, 4].sum(dim=1)

    class FiveChannelPredictor(SequencePredictor):
        def prepare_input(self, values):
            nucleotide_channels = values.transpose(1, 2)
            gc_channel = values[:, :, 1:3].sum(dim=2).unsqueeze(1)
            return torch.cat((nucleotide_channels, gc_channel), dim=1)

    predictor = FiveChannelPredictor(FiveChannelModel())
    discrete = predictor.score(["ACGT", "AAAA"]).scores
    probabilities = torch.from_numpy(onehot(["ACGT", "AAAA"])).requires_grad_()
    soft, _ = predictor.score_soft(probabilities)
    np.testing.assert_array_equal(discrete, [2, 0])
    np.testing.assert_array_equal(soft.detach().squeeze(1), discrete)
    soft.sum().backward()
    assert probabilities.grad is not None
