"""CPU optimizer clients with batched scoring and first-order gradients in one service.

Only the service process owns the prediction model / CUDA context. Optimization
processes communicate NumPy arrays through bounded multiprocessing queues. Soft
backward requests recompute the forward pass, avoiding retained GPU graphs across
processes. The model must be deterministic in eval mode for this recomputation.
"""
from __future__ import annotations

from collections import deque
import csv
import queue
import time
import traceback
from pathlib import Path

import numpy as np

from .api import as_score_batch


class ScoringClient:
    """A model-compatible RPC client; use one client per optimizer process."""
    supports_soft_scoring = True
    device = 'cpu'

    def __init__(self, worker_id, requests, responses, stopped, *, task_names=None,
                 batch_size=512, soft_batch_size=128):
        self.worker_id = worker_id
        self.requests, self.responses, self.stopped = requests, responses, stopped
        self.task_names = task_names
        self.batch_size, self.soft_batch_size = batch_size, soft_batch_size
        self._serial = 0
        self.statistics = dict(requests=0, discrete_rows=0, soft_forward_rows=0,
                               soft_backward_rows=0, rpc_wait_seconds=0.)

    def _call(self, kind, values, gradient=None):
        if self.stopped.is_set():
            raise RuntimeError('Central scoring service stopped')
        serial = self._serial
        self._serial += 1
        started = time.monotonic()
        message = (self.worker_id, serial, kind, values, gradient, started)
        while True:
            try:
                self.requests.put(message, timeout=1.)
                break
            except queue.Full:
                if self.stopped.is_set():
                    raise RuntimeError('Central scoring service stopped while enqueueing')
        while True:
            try:
                received, ok, result = self.responses.get(timeout=1.)
                break
            except queue.Empty:
                if self.stopped.is_set():
                    raise RuntimeError('Central scoring service failed during request')
        if received != serial:
            raise RuntimeError(f'RPC response mismatch: expected {serial}, received {received}')
        if not ok:
            raise RuntimeError(result)
        self.statistics['requests'] += 1
        self.statistics[kind+'_rows'] += len(values)
        self.statistics['rpc_wait_seconds'] += time.monotonic()-started
        return result

    def score(self, sequences):
        values = np.asarray(sequences, dtype=str).reshape(-1)
        batches = [self._call('discrete', values[i:i+self.batch_size])
                   for i in range(0, len(values), self.batch_size)]
        if not batches:
            return np.empty((0, len(self.task_names or ())))
        return np.concatenate(batches)

    def score_soft(self, probabilities):
        import torch
        client = self

        class RemoteActivity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                ctx.save_for_backward(inputs.detach())
                data = inputs.detach().cpu().numpy().copy()
                outputs = [client._call('soft_forward', data[i:i+client.soft_batch_size])
                           for i in range(0, len(data), client.soft_batch_size)]
                return torch.from_numpy(np.concatenate(outputs)).to(inputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, output_gradient):
                (inputs,) = ctx.saved_tensors
                data = inputs.cpu().numpy()
                weights = output_gradient.detach().cpu().numpy()
                gradients = [client._call('soft_backward', data[i:i+client.soft_batch_size],
                                          weights[i:i+client.soft_batch_size])
                             for i in range(0, len(data), client.soft_batch_size)]
                return torch.from_numpy(np.concatenate(gradients)).to(inputs)

        return RemoteActivity.apply(probabilities), None


def scoring_service(factory, requests, responses, stopped, ready, *, max_batch_size=1024,
                    max_soft_batch_size=256, batch_wait_ms=10., metrics_path=None):
    """Own a scorer and merge compatible requests from independent optimizers.

    ``factory`` must construct the scorer inside this process. A single request
    must fit its operation's batch limit; clients split larger calls beforehand.
    ``ready`` receives either ``(True, None)`` or ``(False, traceback)``.
    """
    import torch
    writer = handle = None
    statistics = dict(batches=0, multi_worker_batches=0, discrete_rows=0,
                      soft_forward_rows=0, soft_backward_rows=0,
                      max_discrete_batch=0, max_soft_batch=0, service_seconds=0.)
    pending = deque()
    initialized = False
    try:
        torch.set_num_threads(2)
        scorer = factory()
        device = getattr(scorer, 'device', 'cpu')
        if metrics_path is not None:
            handle = Path(metrics_path).open('w', buffering=1)
            writer = csv.DictWriter(handle, fieldnames=[
                'batch', 'kind', 'rows', 'requests', 'workers',
                'queue_wait_seconds_mean', 'service_seconds'])
            writer.writeheader()
        ready.put((True, None))
        initialized = True
        while True:
            first = pending.popleft() if pending else requests.get()
            if first is None:
                break
            kind = first[2]
            limit = max_batch_size if kind == 'discrete' else max_soft_batch_size
            group = [first]
            size = len(first[3])
            if size > limit:
                raise ValueError(f'{kind} request size {size} exceeds server limit {limit}')
            deadline = time.monotonic()+batch_wait_ms/1000.
            # Inspect already pending requests once; preserve unmatched request order.
            for _ in range(len(pending)):
                current = pending.popleft()
                if current is not None and current[2] == kind and size+len(current[3]) <= limit:
                    group.append(current)
                    size += len(current[3])
                else:
                    pending.append(current)
            while size < limit:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    break
                try:
                    current = requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if current is not None and current[2] == kind and size+len(current[3]) <= limit:
                    group.append(current)
                    size += len(current[3])
                else:
                    pending.append(current)
            began = time.monotonic()
            inputs = np.concatenate([request[3] for request in group])
            if kind == 'discrete':
                outputs = as_score_batch(scorer.score(inputs)).scores
            elif kind in {'soft_forward', 'soft_backward'}:
                values = torch.as_tensor(inputs, dtype=torch.float32, device=device)
                if kind == 'soft_forward':
                    with torch.no_grad():
                        activities, _ = scorer.score_soft(values)
                    outputs = activities.detach().cpu().numpy().copy()
                else:
                    values.requires_grad_(True)
                    activities, _ = scorer.score_soft(values)
                    gradients = torch.as_tensor(np.concatenate([r[4] for r in group]),
                                                dtype=activities.dtype, device=activities.device)
                    (derivative,) = torch.autograd.grad(activities, values, grad_outputs=gradients)
                    outputs = derivative.detach().cpu().numpy().copy()
            else:
                raise ValueError(f'Unknown scoring operation {kind}')
            if len(outputs) != size or not np.isfinite(outputs).all():
                raise ValueError('Central model returned invalid outputs')
            duration = time.monotonic()-began
            offset = 0
            for worker_id, serial, _, values, _, _ in group:
                count = len(values)
                responses[worker_id].put((serial, True, outputs[offset:offset+count].copy()))
                offset += count
            workers = len({r[0] for r in group})
            statistics['batches'] += 1
            statistics['multi_worker_batches'] += workers > 1
            statistics[kind+'_rows'] += size
            maximum = 'max_discrete_batch' if kind == 'discrete' else 'max_soft_batch'
            statistics[maximum] = max(statistics[maximum], size)
            statistics['service_seconds'] += duration
            if writer:
                writer.writerow(dict(batch=statistics['batches'], kind=kind, rows=size,
                    requests=len(group), workers=workers,
                    queue_wait_seconds_mean=sum(began-r[5] for r in group)/len(group),
                    service_seconds=duration))
        if metrics_path:
            import json
            Path(metrics_path).with_suffix('.json').write_text(json.dumps(statistics, indent=2)+'\n')
    except BaseException:
        failure = traceback.format_exc()
        if not initialized:
            ready.put((False, failure))
        if metrics_path:
            Path(metrics_path).with_suffix('.error.txt').write_text(failure)
        stopped.set()
        raise
    finally:
        if handle:
            handle.close()
