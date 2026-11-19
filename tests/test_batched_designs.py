"""High-level generation uses real CPU processes and one shared model service."""
import multiprocessing as mp
import os
import threading

import numpy as np
import pandas as pd
import pytest
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
        self.calls.append((os.getpid(), threading.current_thread().name, len(x)))
        return x[:, self.channel].sum(1, keepdim=True)


def test_default_queue_batching_preserves_serial_designs_and_model_ownership(tmp_path):
    model = TrackedModel()
    initial = random_seed_sequences(12, 4, seed=7)
    options = dict(length=12, initial=initial, objective=['maximize'],
        algorithms=['basesearch', 'genetic_algorithm', 'fast_seqprop'],
        num_candidates=4, score_threshold=9, diversity=1, seed=8)
    serial = generate_designs(model, **options, parallel=False)
    model.calls.clear()
    parallel = generate_designs(model, **options, output_dir=tmp_path/'designs',
        max_batch_size=16, max_soft_batch_size=4, batch_wait_ms=30.)
    pd.testing.assert_frame_equal(parallel, serial)
    assert parallel.attrs['execution'] == 'parallel'
    assert len({r['pid'] for r in parallel.attrs['timing']}) == 3
    assert all(r['pid'] != os.getpid() for r in parallel.attrs['timing'])
    assert {pid for pid, _, _ in model.calls} == {os.getpid()}
    assert any(thread == 'bioseq-scoring-service' for _, thread, _ in model.calls)
    stats = parallel.attrs['batching']
    assert stats['multi_worker_batches'] > 0
    assert stats['max_discrete_batch'] <= 16
    assert stats['max_soft_batch'] <= 4
    assert stats['soft_forward_rows'] > 0 and stats['soft_backward_rows'] > 0
    batches = pd.read_csv(tmp_path/'designs/gpu_batches.csv')
    assert (batches.workers > 1).any()
    assert set(batches.kind) == {'discrete', 'soft_forward', 'soft_backward'}
    assert (tmp_path/'designs/gpu_batches.json').exists()


def test_combined_local_models_keep_high_low_objective_and_gradients():
    models = [TrackedModel(2), TrackedModel(1), TrackedModel(0)]
    scorer = CombinedScorer([SequencePredictor(m) for m in models],
        names=['target', 'off1', 'off2'])
    objective = MultiTaskObjective(high=['target'], low=['off1', 'off2'])
    designs = generate_designs(scorer, 12, num_candidates=3,
        algorithms=['genetic_algorithm', 'fast_seqprop'], score_threshold=0,
        diversity=1, objective=objective, seed=12, batch_wait_ms=20.)
    expected = (designs.activity_target-designs.activity_off1-designs.activity_off2)/3
    np.testing.assert_allclose(designs.score, expected)
    assert designs.attrs['task_names'] == ['target', 'off1', 'off2']
    assert designs.attrs['batching']['soft_backward_rows'] > 0
    assert all({pid for pid, _, _ in m.calls} == {os.getpid()} for m in models)


def test_worker_failure_is_reported_and_processes_are_cleaned_up():
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(ValueError, match="fixed_region positions"):
        generate_designs(TrackedModel(), 8, num_candidates=4,
            algorithms=['random', 'genetic_algorithm'],
            initial=['AAAAAAAA', 'CCCCCCCC', 'GGGGGGGG', 'TTTTTTTT'], fixed_region=[0])
    assert {p.pid for p in mp.active_children()} == before
    assert not any(t.name == 'bioseq-scoring-service' for t in threading.enumerate())


def test_model_failure_stops_waiting_optimizers():
    def fail(sequences):
        raise ValueError('deliberate model failure')
    before = {p.pid for p in mp.active_children()}
    scorer = CallableScorer(fail, task_names=['activity'])
    with pytest.raises(RuntimeError, match='scoring service failed') as error:
        generate_designs(scorer, 8, num_candidates=2,
            algorithms=['random', 'genetic_algorithm'], score_threshold=0)
    assert isinstance(error.value.__cause__, ValueError)
    assert {p.pid for p in mp.active_children()} == before
    assert not any(t.name == 'bioseq-scoring-service' for t in threading.enumerate())


def test_batched_score_responses_preserve_uncertainties(tmp_path):
    from sequence_optimization.parallel import run_batched_designs
    def score(sequences):
        values = np.array([s.count('G') for s in sequences], dtype=float)
        return values, np.full(len(sequences), .25)
    scorer = CallableScorer(score, task_names=['activity'])
    completed, _ = run_batched_designs(scorer, None, 4,
        np.array(['AAAA', 'CCCC', 'GGGG', 'TTTT']), ['maximize'], 2,
        [('random', dict(score_threshold=0, diversity=0))], max_batch_size=2,
        metrics_path=str(tmp_path/'batches.csv'))
    result, _ = completed[0]
    np.testing.assert_allclose(result.uncertainties, .25)


def test_separate_differentiable_model_routes_its_discrete_predictions():
    discrete = SequencePredictor(TrackedModel(2), tasks=['activity'])
    gradient = SequencePredictor(TrackedModel(1), tasks=['activity'])
    options = dict(length=8, initial=['AAAAAAAA', 'CCCCCCCC', 'GGGGGGGG', 'TTTTTTTT'],
        objective=['maximize'], algorithms=['random', 'fast_seqprop'],
        differentiable_model=gradient, num_candidates=2, score_threshold=0, diversity=0,
        seed=3)
    serial = generate_designs(discrete, **options, parallel=False)
    parallel = generate_designs(discrete, **options, batch_wait_ms=20.)
    pd.testing.assert_frame_equal(parallel, serial)
    assert parallel.attrs['batching']['discrete_gradient_rows'] > 0
    for row in parallel.itertuples():
        expected = row.sequence.count('C' if row.algorithm == 'fast_seqprop' else 'G')
        assert row.score == expected
