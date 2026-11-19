from tests.helpers import run_kernel

import numpy as np
import pytest
import torch

from sequence_optimization import (
    CallableScorer,
    SequencePredictor,
    OptimizationConfig,
    SequenceSpace,
)


def make_scorer():
    class GCount(torch.nn.Module):
        def forward(self, inputs):
            # The model must see categorical samples, including during backward.
            assert torch.all((inputs.detach() == 0) | (inputs.detach() == 1))
            return inputs[:, :, 2].sum(dim=1)

    class ChannelsLastPredictor(SequencePredictor):
        def prepare_input(self, values):
            return values

    return ChannelsLastPredictor(GCount(), task_names=("g",))


@pytest.mark.parametrize("budget", [4, 13, 84])
def test_ledidi_budget_reproducibility_and_loss(budget):
    options = dict(
        initial=["AAAAAAAA", "CCCCCCCC", "TTTTTTTT", "AAAACCCC"],
        objective=["maximize"],
        num_candidates=4,
        diversity=0,
        budget=budget,
        seed=7,
        learning_rate=1.0,
        ledidi_epsilon=0.2,
        ledidi_edit_penalty=0.05,
    )
    scorer = make_scorer()
    first = run_kernel(scorer, SequenceSpace(8), algorithm="ledidi", **options)
    second = run_kernel(scorer, SequenceSpace(8), algorithm="ledidi", **options)
    np.testing.assert_array_equal(first.sequences, second.sequences)
    assert first.query_count == budget // 4 * 4
    assert first.metadata["soft_query_count"] == first.query_count - 4
    assert len(set(first.sequences)) == 4
    np.testing.assert_allclose(first.scores, first.activities[:, 0])
    for step in first.trajectory:
        assert step["loss"] == pytest.approx(
            -step["mean_score"] + 0.05 * step["mean_edits"], abs=1e-5
        )
    if budget == 84:
        assert first.scores.max() > 0


@pytest.mark.parametrize(
    "option,value",
    [
        ("ledidi_edit_penalty", -1),
        ("ledidi_edit_penalty", float("nan")),
        ("ledidi_temperature", 0),
        ("ledidi_temperature", float("inf")),
        ("ledidi_epsilon", -1),
    ],
)
def test_ledidi_invalid_config(option, value):
    with pytest.raises(ValueError, match=option):
        OptimizationConfig(4, num_candidates=4, **{option: value})


def test_ledidi_requires_gradients():
    with pytest.raises(TypeError, match="score_soft"):
        run_kernel(
            CallableScorer(lambda x: np.ones(len(x))),
            SequenceSpace(8),
            None,
            ["maximize"],
            4,
            algorithm='ledidi', budget=8,
        )


def test_ledidi_accepts_multitask_objective():
    from sequence_optimization import MultiTaskObjective

    class MultiTask:
        def score(self, values):
            return np.ones((len(values), 2))

        def score_soft(self, values):
            return values[:, :, :2].mean(dim=1), None

    result = run_kernel(
        MultiTask(),
        SequenceSpace(8),
        initial=None,
        algorithm='ledidi', budget=8,
        objective=MultiTaskObjective(high="all"),
        num_candidates=4,
    )
    assert result.activities.shape == (4, 2)
    assert result.metadata["soft_query_count"] == 4
