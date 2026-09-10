from functools import partial

import numpy as np
import pytest
import torch

from sequence_optimization import (
    DifferentiableSequencePredictor,
    DifferentiableTorchSequenceScorer,
    SequencePredictor,
    TorchSequenceScorer,
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
    predictor = make_predictor(MyModel(), differentiable=True)
    inputs = torch.from_numpy(onehot(["AGGT", "AAAA"])).requires_grad_()
    scores, _ = predictor.score_soft(inputs)
    np.testing.assert_allclose(
        scores.detach(), predictor.score(["AGGT", "AAAA"]).scores
    )
    scores.sum().backward()
    np.testing.assert_allclose(inputs.grad[:, :, 2], 0.25)
    assert inputs.grad[:, :, [0, 1, 3]].eq(0).all()


def test_predictor_names_and_conflicting_converters():
    assert TorchSequenceScorer is SequencePredictor
    assert DifferentiableTorchSequenceScorer is DifferentiableSequencePredictor
    with pytest.raises(ValueError, match="one input"):
        SequencePredictor(MyModel(), onehot, onehot=partial(onehot, alphabet="AGCT"))
