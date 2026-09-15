import numpy as np
import pytest
import torch

from sequence_optimization import (
    CallableScorer,
    DifferentiableTorchSequenceScorer,
    OptimizationConfig,
    SequenceSpace,
    ledidi_optimize,
)


def make_scorer():
    class GCount(torch.nn.Module):
        def forward(self, inputs):
            # The model must see categorical samples, including during backward.
            assert torch.all((inputs.detach() == 0) | (inputs.detach() == 1))
            return inputs[:, :, 2].sum(dim=1)

    def encode(values):
        return torch.nn.functional.one_hot(
            torch.as_tensor(SequenceSpace(8).encode(values), dtype=torch.long), 4
        ).float()

    return DifferentiableTorchSequenceScorer(GCount(), encode, task_names=("g",))


@pytest.mark.parametrize("budget", [4, 13, 84])
def test_ledidi_budget_reproducibility_and_loss(budget):
    options = dict(
        initial=["AAAAAAAA", "CCCCCCCC", "TTTTTTTT", "AAAACCCC"],
        num_candidates=4,
        diversity=0,
        budget=budget,
        seed=7,
        learning_rate=1.0,
        ledidi_epsilon=0.2,
        ledidi_edit_penalty=0.05,
    )
    scorer = make_scorer()
    first = ledidi_optimize(scorer, SequenceSpace(8), **options)
    second = ledidi_optimize(scorer, SequenceSpace(8), **options)
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
        ledidi_optimize(
            CallableScorer(lambda x: np.ones(len(x))),
            SequenceSpace(8),
            budget=8,
            num_candidates=4,
        )


def test_ledidi_accepts_multitask_objective():
    from sequence_optimization import MultiTaskObjective

    class MultiTask:
        def score(self, values):
            return np.ones((len(values), 2))

        def score_soft(self, values):
            return values[:, :, :2].mean(dim=1), None

    result = ledidi_optimize(
        MultiTask(),
        SequenceSpace(8),
        budget=8,
        num_candidates=4,
        objective=MultiTaskObjective(high="all"),
    )
    assert result.activities.shape == (4, 2)
    assert result.metadata["soft_query_count"] == 4
