"""Verify that independent runs submit the same scoring requests in order."""

import hashlib
import os
import threading

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from sequence_optimization import (
    CallableScorer, CombinedScorer, MultiTaskObjective, SequencePredictor,
    generate_designs, random_seed_sequences,
)


class TrackedModel(nn.Module):
    def __init__(self, channel=2):
        super().__init__()
        self.channel = channel
        self.calls = []

    def forward(self, x):
        digest = hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
        self.calls.append((os.getpid(), threading.get_ident(), x.requires_grad, digest))
        return x[:, self.channel].sum(1, keepdim=True)


def test_default_execution_repeats_scoring_requests_and_candidate_order(tmp_path, capsys):
    model = TrackedModel()
    initial = random_seed_sequences(12, 4, seed=7)
    options = dict(length=12, initial=initial, objective=["maximize"],
                   num_candidates=4, score_threshold=9, diversity=1, seed=8)
    first = generate_designs(model, **options, output_dir=tmp_path / "first")
    first_calls = model.calls.copy()
    model.calls.clear()
    second = generate_designs(model, **options, output_dir=tmp_path / "second")
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first_calls == model.calls
    assert any(requires_grad for _, _, requires_grad, _ in model.calls)
    assert {(pid, thread) for pid, thread, _, _ in model.calls} == {
        (os.getpid(), threading.get_ident())
    }
    assert first.algorithm.nunique() == 11
    assert first.attrs["execution"] == "serial"
    assert first.algorithm.drop_duplicates().tolist() == first.attrs["algorithms"]
    assert [row["seed"] for row in first.attrs["timing"]] == list(range(8, 19))
    assert (tmp_path / "first/designs.csv").read_bytes() == (tmp_path / "second/designs.csv").read_bytes()
    assert "11/11 optimizers complete" in capsys.readouterr().err


def test_combined_models_keep_high_low_objective_and_gradients():
    models = [TrackedModel(2), TrackedModel(1), TrackedModel(0)]
    scorer = CombinedScorer([SequencePredictor(m) for m in models],
                            names=["target", "off1", "off2"])
    objective = MultiTaskObjective(high=["target"], low=["off1", "off2"])
    designs = generate_designs(scorer, 12, num_candidates=3,
        algorithms=["genetic_algorithm", "fast_seqprop"], score_threshold=0,
        diversity=1, objective=objective, seed=12)
    expected = (designs.activity_target - designs.activity_off1 - designs.activity_off2) / 3
    np.testing.assert_allclose(designs.score, expected)
    assert designs.attrs["task_names"] == ["target", "off1", "off2"]
    assert all(any(call[2] for call in m.calls) for m in models)


def test_model_failure_restores_torch_settings():
    before = (torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)

    def fail(sequences):
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cudnn.benchmark
        raise ValueError("deliberate model failure")

    scorer = CallableScorer(fail, task_names=["activity"])
    with pytest.raises(ValueError, match="deliberate model failure"):
        generate_designs(scorer, 8, num_candidates=2,
                         algorithms=["random", "genetic_algorithm"], score_threshold=0)
    after = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    assert before == after


def test_separate_differentiable_model_routes_its_discrete_predictions():
    discrete = SequencePredictor(TrackedModel(2), tasks=["activity"])
    gradient = SequencePredictor(TrackedModel(1), tasks=["activity"])
    designs = generate_designs(
        discrete, length=8, initial=["AAAAAAAA", "CCCCCCCC", "GGGGGGGG", "TTTTTTTT"],
        objective=["maximize"], algorithms=["random", "fast_seqprop"],
        differentiable_model=gradient, num_candidates=2, score_threshold=0, diversity=0,
        seed=3,
    )
    for row in designs.itertuples():
        expected = row.sequence.count("C" if row.algorithm == "fast_seqprop" else "G")
        assert row.score == expected
