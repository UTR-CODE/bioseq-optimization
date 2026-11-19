"""CPU optimizer clients with batched scoring and first-order gradients in one service.

Only the scoring service owns the prediction model / CUDA context. Optimization
processes communicate NumPy arrays through bounded multiprocessing queues. Soft
backward requests recompute the forward pass, avoiding retained GPU graphs across
processes. The model must be deterministic in eval mode for this recomputation.
"""
from __future__ import annotations

from collections import deque
import csv
import math
import queue
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .api import ScoreBatch, as_score_batch


class ScoringClient:
    """A model-compatible RPC client; use one client per optimizer process."""
    supports_soft_scoring = True
    device = 'cpu'

    def __init__(self, worker_id, requests, responses, stopped, *, task_names=None,
                 batch_size=512, soft_batch_size=128, discrete_kind='discrete'):
        self.worker_id = worker_id
        self.requests, self.responses, self.stopped = requests, responses, stopped
        self.task_names = task_names
        self.batch_size, self.soft_batch_size = batch_size, soft_batch_size
        self.discrete_kind = discrete_kind
        self._serial = 0
        self.statistics = dict(requests=0, discrete_rows=0, discrete_gradient_rows=0, soft_forward_rows=0,
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
        batches = [self._call(self.discrete_kind, values[i:i+self.batch_size])
                   for i in range(0, len(values), self.batch_size)]
        if not batches:
            return np.empty((0, len(self.task_names or ())))
        if all(isinstance(batch, np.ndarray) for batch in batches):
            return np.concatenate(batches)
        batches = [as_score_batch(batch) for batch in batches]
        uncertainty = None if any(batch.uncertainties is None for batch in batches) else np.concatenate(
            [batch.uncertainties for batch in batches])
        return ScoreBatch(np.concatenate([batch.scores for batch in batches]), uncertainty)

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

    ``factory`` supplies the scorer inside the service process or thread. A request
    must fit its operation's batch limit; clients split larger calls beforehand.
    ``ready`` receives either ``(True, None)`` or ``(False, traceback)``.
    """
    import torch
    writer = handle = None
    statistics = dict(batches=0, multi_worker_batches=0, discrete_rows=0, discrete_gradient_rows=0,
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
        while not stopped.is_set():
            try:
                first = pending.popleft() if pending else requests.get(timeout=1.)
            except queue.Empty:
                continue
            if first is None:
                break
            kind = first[2]
            is_discrete = kind in {'discrete', 'discrete_gradient'}
            limit = max_batch_size if is_discrete else max_soft_batch_size
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
            uncertainty = None
            if is_discrete:
                predict = scorer.score if kind == 'discrete' else scorer.score_gradient
                batch = as_score_batch(predict(inputs))
                outputs, uncertainty = batch.scores, batch.uncertainties
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
                result = outputs[offset:offset+count].copy()
                if uncertainty is not None:
                    result = ScoreBatch(result, uncertainty[offset:offset+count].copy())
                responses[worker_id].put((serial, True, result))
                offset += count
            workers = len({r[0] for r in group})
            statistics['batches'] += 1
            statistics['multi_worker_batches'] += workers > 1
            statistics[kind+'_rows'] += size
            maximum = 'max_discrete_batch' if is_discrete else 'max_soft_batch'
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


class _DesignScorer:
    """Keep the shared discrete cache and differentiable model in the caller."""
    def __init__(self, scorer, gradient_scorer):
        self.scorer, self.gradient_scorer = scorer, gradient_scorer
        self.device = getattr(gradient_scorer, 'device', None) or 'cpu'

    def score(self, sequences):
        return self.scorer.score(sequences)

    def score_gradient(self, sequences):
        return self.gradient_scorer.score(sequences)

    def score_soft(self, probabilities):
        return self.gradient_scorer.score_soft(probabilities)


class _DesignProgress:
    """Refresh a terminal panel or write periodic snapshots to a log."""

    def __init__(self, names, state):
        self.names = names
        self.state = state
        self.stream = sys.stderr
        self.live = self.stream.isatty()
        self.started = time.monotonic()
        self.next_refresh = self.started
        self.rendered = False

    def refresh(self, completed, *, force=False):
        now = time.monotonic()
        if not force and now < self.next_refresh:
            return
        self.next_refresh = now + 5
        count, total = len(completed), len(self.names)
        filled = 20 * count // total
        bar = "#" * filled + "-" * (20 - filled)
        lines = [f"[{bar}] {count}/{total} 个算法已完成  |  {now - self.started:.0f} s"]
        state = self.state[:]
        for index, name in enumerate(self.names):
            round_number, mean = state[2 * index:2 * index + 2]
            status = "已完成" if index in completed else "优化中"
            score = f"第 {int(round_number)} 轮  目标均分 {mean:.4f}" if math.isfinite(mean) else "等待评分"
            if index in completed and not math.isfinite(mean):
                score = ""
            lines.append(f"  {name:32} {status}  {score}".rstrip())
        if self.live:
            if self.rendered:
                self.stream.write(f"\x1b[{len(lines)}F")
            lines = ["\x1b[2K" + line for line in lines]
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self.rendered = True


def _design_worker(index, client, length, initial, score_settings, count, name,
                   alphabet, seed, options, barrier, results, progress_state):
    """CPU-only proposals and remote scoring; no model is sent to this process."""
    import os
    import cloudpickle
    import torch
    import numba
    from .algorithms import optimize
    torch.set_num_threads(1)
    numba.set_num_threads(1)
    if progress_state is not None:
        def report_progress(round_number, mean):
            progress_state[2 * index:2 * index + 2] = (round_number, mean)
        client.report_progress = report_progress
    try:
        objective, constraint = cloudpickle.loads(score_settings)
        barrier.wait()
        began, cpu = time.perf_counter(), time.process_time()
        result = optimize(client, length, initial, objective, count, algorithm=name,
                          alphabet=alphabet, constraint=constraint, seed=seed, **options)
        results.put((index, True, result, dict(seconds=time.perf_counter()-began,
            cpu_seconds=time.process_time()-cpu, pid=os.getpid())))
    except BaseException as error:
        results.put((index, False, error, traceback.format_exc()))


def run_batched_designs(scorer, gradient_scorer, length, initial, objective, count,
                        jobs, *, alphabet='ACGT', constraint=None, seed=42,
                        max_batch_size=512, max_soft_batch_size=64,
                        batch_wait_ms=5., metrics_path=None, show_progress=True):
    """Run independent optimizers through one local, model-owning service thread.

    Only clients, sequences and optimization options cross process boundaries.
    Keeping inference in the caller supports already-loaded CUDA models and
    notebook-defined PyTorch classes without serializing model objects.
    """
    import json
    import multiprocessing as mp
    import os
    import threading
    import cloudpickle
    import torch

    # Notebook-defined objectives and constraints must cross the spawn boundary.
    score_settings = cloudpickle.dumps((objective, constraint))
    # Infer names once; the same central cache serves all subsequent requests.
    if getattr(scorer, 'task_names', None) is None:
        scorer.score(initial[:1])
    task_names = getattr(scorer, 'task_names', None)
    ctx = mp.get_context('spawn')
    requests, ready, results = ctx.Queue(maxsize=2*len(jobs)), ctx.Queue(), ctx.Queue()
    replies = [ctx.Queue(maxsize=2) for _ in jobs]
    stopped = ctx.Event()
    barrier = ctx.Barrier(len(jobs))
    failures = []
    combined = _DesignScorer(scorer, gradient_scorer)

    def serve():
        try:
            scoring_service(lambda: combined, requests, replies, stopped, ready,
                max_batch_size=max_batch_size, max_soft_batch_size=max_soft_batch_size,
                batch_wait_ms=batch_wait_ms, metrics_path=metrics_path)
        except BaseException as error:
            failures.append(error)

    service = threading.Thread(target=serve, name='bioseq-scoring-service', daemon=True)
    workers, received = [], {}
    # Each worker writes only its own pair. A progress read must not block if
    # a worker is killed while publishing, since the parent also checks exits.
    progress_state = ctx.RawArray("d", [0.0, math.nan] * len(jobs)) if show_progress else None
    progress = _DesignProgress([name for name, _ in jobs], progress_state) if show_progress else None
    previous_threads = torch.get_num_threads()
    thread_env = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    previous_env = {key: os.environ.get(key) for key in thread_env}
    try:
        service.start()
        ok, error = ready.get()
        if not ok:
            raise RuntimeError(error)
        # Spawned optimizers each use one CPU thread, including imported BLAS.
        for key in thread_env:
            os.environ[key] = '1'
        from .suite import DIFFERENTIABLE_ALGORITHMS
        for index, (name, options) in enumerate(jobs):
            client = ScoringClient(index, requests, replies[index], stopped,
                task_names=task_names, batch_size=max_batch_size,
                soft_batch_size=max_soft_batch_size,
                discrete_kind='discrete_gradient' if name in DIFFERENTIABLE_ALGORITHMS
                    and gradient_scorer is not scorer else 'discrete')
            process = ctx.Process(target=_design_worker,
                name=f'bioseq-{name}', args=(index, client, length, initial, score_settings,
                    count, name, alphabet, seed+index, options, barrier, results,
                    progress_state))
            process.start()
            workers.append(process)
        while len(received) < len(jobs):
            if progress is not None:
                progress.refresh(received)
            try:
                index, ok, result, info = results.get(timeout=.1)
            except queue.Empty:
                if not service.is_alive() or failures:
                    raise RuntimeError('Shared scoring service failed') from (failures[0] if failures else None)
                failed = next((p for p in workers if p.exitcode not in (None, 0)), None)
                if failed is not None:
                    raise RuntimeError(f'Optimizer {failed.name} exited with code {failed.exitcode}')
                continue
            if not ok:
                if failures:
                    raise RuntimeError('Shared scoring service failed') from failures[0]
                if hasattr(result, 'add_note'):
                    result.add_note(info)
                raise result
            received[index] = (result, info)
        if progress is not None:
            progress.refresh(received, force=True)
        requests.put(None)
        service.join()
        if failures:
            raise RuntimeError('Shared scoring service failed') from failures[0]
        for process in workers:
            process.join()
        statistics = json.loads(Path(metrics_path).with_suffix('.json').read_text()) if metrics_path else {}
        return [received[i] for i in range(len(jobs))], statistics
    finally:
        stopped.set()
        if service.is_alive():
            try:
                requests.put_nowait(None)
            except queue.Full:
                pass
            service.join(timeout=5.)
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.)
        for channel in [requests, ready, results, *replies]:
            channel.cancel_join_thread()
            channel.close()
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        torch.set_num_threads(previous_threads)
