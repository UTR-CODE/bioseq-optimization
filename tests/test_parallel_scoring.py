"""Check cross-process batching, response routing, and remote autograd."""
import multiprocessing as mp
import time

import numpy as np
import pytest
import torch

from sequence_optimization.parallel import ScoringClient, scoring_service


class ToyScorer:
    device = 'cpu'
    def score_soft(self, p):
        weights = torch.arange(1, 5, dtype=p.dtype, device=p.device)
        return torch.stack(((p.square()*weights).sum((1, 2)), p.sum((1, 2))), dim=1), None
    def score(self, sequences):
        x = np.zeros((len(sequences), 4, 4), dtype=np.float32)
        for i, s in enumerate(sequences):
            for j, base in enumerate(s):
                x[i, j, 'ACGT'.index(base)] = 1
        with torch.no_grad():
            return self.score_soft(torch.from_numpy(x))[0].numpy()


def fail_factory():
    raise ValueError('deliberate factory failure')


def exercise_client(client, barrier, results):
    try:
        torch.set_num_threads(1)
        barrier.wait(timeout=20)
        sequences = np.array(['ACGT', 'AAAA', 'CCCC', 'GGGG'])
        scores = client.score(sequences)
        np.testing.assert_allclose(scores, ToyScorer().score(sequences))
        p = torch.tensor(np.random.default_rng(client.worker_id).random((4, 4, 4)),
                         dtype=torch.float32, requires_grad=True)
        output, _ = client.score_soft(p)
        weights = torch.tensor([[.5, .2], [.1, -.3], [.9, .2], [-.2, .7]])
        (gradient,) = torch.autograd.grad(output, p, weights)
        direct, _ = ToyScorer().score_soft(p)
        (expected,) = torch.autograd.grad(direct, p, weights)
        np.testing.assert_allclose(output.detach(), direct.detach(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-6)
        results.put(None)
    except Exception as error:
        results.put(repr(error))
        raise


def test_process_batching_and_gradients(tmp_path):
    torch.set_num_threads(1)
    ctx = mp.get_context('spawn')
    requests, ready, results = ctx.Queue(12), ctx.Queue(), ctx.Queue()
    replies = [ctx.Queue() for _ in range(3)]
    stopped = ctx.Event()
    service = ctx.Process(target=scoring_service, args=(ToyScorer, requests, replies, stopped, ready),
        kwargs=dict(max_batch_size=8, max_soft_batch_size=8, batch_wait_ms=100,
                    metrics_path=str(tmp_path/'batches.csv')))
    clients = [ScoringClient(i, requests, replies[i], stopped, task_names=('x','y'),
                            batch_size=3, soft_batch_size=3) for i in range(3)]
    barrier = ctx.Barrier(3)
    service.start()
    workers = []
    try:
        assert ready.get(timeout=20)[0]
        workers = [ctx.Process(target=exercise_client, args=(c, barrier, results)) for c in clients]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
        assert [results.get(timeout=2) for _ in workers] == [None]*3
        requests.put(None)
        service.join(timeout=10)
        assert service.exitcode == 0
        import json
        stats = json.loads((tmp_path/'batches.json').read_text())
        assert stats['multi_worker_batches'] > 0
        assert stats['max_discrete_batch'] <= 8 and stats['max_soft_batch'] <= 8
        assert stats['discrete_rows'] == stats['soft_forward_rows'] == stats['soft_backward_rows'] == 12
    finally:
        for process in [*workers, service]:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_initialization_failure_is_reported():
    ctx = mp.get_context('spawn')
    requests, ready = ctx.Queue(), ctx.Queue()
    stopped = ctx.Event()
    service = ctx.Process(target=scoring_service, args=(fail_factory, requests, [], stopped, ready))
    service.start()
    try:
        ok, error = ready.get(timeout=20)
        assert not ok and 'deliberate factory failure' in error
        service.join(timeout=5)
        assert stopped.is_set() and service.exitcode != 0
        client = ScoringClient(0, requests, None, stopped)
        with pytest.raises(RuntimeError, match='stopped'):
            client.score(['AAAA'])
    finally:
        if service.is_alive():
            service.terminate()
            service.join(timeout=5)
