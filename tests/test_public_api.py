"""Test user-facing adapters and objectives through the public workflow."""

import inspect
import json

import numpy as np
import pytest
import torch
from torch import nn

from MotifSift import CombinedScorer, MultiTaskObjective, SequencePredictor, generate_designs
from sequence_optimization.attribution import ism_contributions
from sequence_optimization.objectives import ObjectiveScorer


class Counts(nn.Module):
    def forward(self, x):
        return x[:, :3].sum(dim=2)


class ChannelsLastModel(nn.Module):
    def forward(self, x):
        return x[:, :, 1].sum(dim=1)


class ChannelsLastPredictor(SequencePredictor):
    def prepare_input(self, probabilities):
        return probabilities


def test_model_adapters_and_named_weights():
    assert list(inspect.signature(SequencePredictor).parameters) == ["model", "tasks"]
    assert "parallel" not in inspect.signature(generate_designs).parameters
    with pytest.raises(TypeError, match="SequencePredictor"):
        generate_designs(Counts(), length=8)
    scorer = ChannelsLastPredictor(ChannelsLastModel(), ["activity"])
    output, _ = scorer.score_soft(torch.eye(4)[None])
    np.testing.assert_allclose(output.detach(), scorer.score(["ACGT"]).scores)
    combined = CombinedScorer({"target": scorer, "off": scorer})
    designs = generate_designs(
        combined, length=8, initial=["AAACCCAA", "GGGCCCCT"],
        num_candidates=2, algorithms=["basesearch"], score_threshold=0,
        objective=MultiTaskObjective(high=["target"], low=["off"], weights={"target": 2}),
    )
    np.testing.assert_allclose(designs.score, designs.activity_target / 3)
    assert designs.attrs["execution"] == "parallel"


def test_notebook_style_custom_objective_runs_with_gradients_and_ism(tmp_path):
    scorer = SequencePredictor(Counts(), ["target", "off1", "off2"])
    multiplier = 2.0

    def specificity(y):
        return multiplier * y["target"] - torch.maximum(y["off1"], y["off2"])

    def without_homopolymers(sequence):
        return len(set(sequence)) > 1

    designs = generate_designs(
        scorer, length=8, initial=["ACTCCTAT", "GGTAATTA"], num_candidates=2,
        algorithms=["basesearch", "fast_seqprop"], score_threshold=9,
        objective=specificity, constraint=without_homopolymers, output_dir=tmp_path,
    )
    assert set(designs.algorithm) == {"basesearch", "fast_seqprop"}
    assert designs.attrs["batching"]["soft_backward_rows"] > 0
    np.testing.assert_allclose(designs.score, 2 * designs.activity_target - np.maximum(
        designs.activity_off1, designs.activity_off2,
    ))
    assert json.loads((tmp_path / "config.json").read_text())["objective"] == {
        "type": "custom", "function": "specificity",
    }
    objective_scorer = ObjectiveScorer(scorer, designs.attrs["objective"])
    result = ism_contributions(["ACTCCTAT"], lambda batch: objective_scorer.score(batch).scores)
    assert result.reference_predictions[0] == 1


def test_single_output_minimization_uses_all_default_optimizers():
    scorer = ChannelsLastPredictor(ChannelsLastModel(), ["activity"])
    designs = generate_designs(
        scorer, length=8, objective="minimize", num_candidates=2,
        initial=["AAACCCAA", "GGGCCCCT"], score_threshold=-8,
    )
    assert designs.algorithm.nunique() == 11
    assert designs.groupby("algorithm").size().eq(2).all()
    np.testing.assert_allclose(designs.score, -designs.activity_activity)


def test_custom_scorer_receives_normalized_lists_and_accepts_a_soft_tensor():
    class ListScorer:
        def score(self, sequences):
            assert type(sequences) is list
            assert all(set(sequence) <= set('ACGTN') for sequence in sequences)
            return [sequence.count('G') for sequence in sequences]

        def score_soft(self, probabilities):
            assert probabilities.shape[1:] == (8, 4)
            return probabilities[:, :, 2].sum(dim=1)

    model = ListScorer()
    designs = generate_designs(
        model, length=8, initial=['ACUCCUAU', 'GGUAAUUA'], num_candidates=2,
        algorithms=['basesearch', 'fast_seqprop'], score_threshold=6,
    )
    assert designs.attrs['batching']['soft_backward_rows'] > 0
    assert all(set(sequence) <= set('ACGT') for sequence in designs.sequence)
    np.testing.assert_array_equal(designs.score, designs.sequence.str.count('G'))

    combined = CombinedScorer({'activity': model})
    np.testing.assert_array_equal(combined.score(['guN']).scores, [[1]])
    with pytest.raises(ValueError, match='contain only A, C, G, T, U, or N'):
        combined.score(['GUX'])


def test_default_optimizer_seed_matches_explicit_seed_42():
    model = ChannelsLastPredictor(ChannelsLastModel(), ["activity"])
    options = dict(length=8, num_candidates=2, algorithms=["basesearch", "genetic_algorithm"], score_threshold=4)
    default = generate_designs(model, **options)
    explicit = generate_designs(model, seed=42, **options)
    assert default.equals(explicit)
    assert default.attrs["seed"] == 42
    assert [entry["seed"] for entry in default.attrs["timing"]] == [42, 43]
