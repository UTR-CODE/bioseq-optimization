"""Run all 13 algorithms on one task and save results to disk.

Usage:
    python run_task.py <task_index>

where <task_index> is a zero-based index into the TASKS list in tasks.py.
"""
import sys
import os
import time
import traceback
import pickle

sys.path.insert(0, '/home/hexin/bioseq-models')
sys.path.insert(0, '/mnt/data/home/hexin/MPRA_benchmark')

import numpy as np

from config import (
    RESULTS_DIR, SEED, NUM_INITIAL, NUM_RETURN,
    ALGORITHMS, GRADIENT_ALGORITHMS,
)
from sequence_optimization import (
    random_seed_sequences, optimize, SaturationConfig,
)
from sequence_optimization.saturation import SaturationResult


def result_path(task_id, algo):
    return os.path.join(RESULTS_DIR, f'{task_id}__{algo}.pkl')


def run_one(task, algo):
    """Run a single algorithm on a task; return a result dict."""
    scorer    = task['make_scorer']()
    objective = task['objective']
    length    = task['length']
    is_cds    = task['is_cds']

    # Stop at plateau with a hard ceiling from the task's max_queries field.
    # Explicit minimum/patience avoid the default 10× and 20× num_candidates
    # multipliers that would require 30M queries for num_candidates=1000.
    max_q  = task.get('max_queries', 100_000)
    min_q  = min(5_000, max_q // 10)    # at least see the initial population a few times
    pat_q  = min(5_000, max_q // 10)    # patience before calling plateau
    saturation = SaturationConfig(
        fractions=(0.99,),
        minimum_queries=min_q,
        plateau_patience_queries=pat_q,
        maximum_queries=max_q,
        material_improvement=1e-4,
    )

    # diversity=0 disables the minimum-Hamming-distance filter. Required here:
    # SaturationMonitor.observe() runs _select_diverse() on every scoring batch,
    # which is O(archive x num_candidates) pairwise comparisons in Python and
    # dominates runtime at num_candidates=1000. It also matches the
    # "no sequence constraints" setting for this benchmark.
    # random search never plateaus (each fresh batch keeps improving the
    # top-1000 archive by more than material_improvement), so the library
    # rejects saturation for it and requires an explicit budget. Give it the
    # same query ceiling as the other algorithms to keep the comparison fair.
    if algo == 'random':
        stop_kwargs = {'budget': max_q}
    else:
        stop_kwargs = {'saturation': saturation}

    if is_cds:
        # Pass the single reference CDS as initial.
        # optimize() with is_CDS_optimized=True builds a SynonymousCodonSpace
        # and draws num_candidates synonymous variants from it internally.
        result = optimize(
            scorer, length,
            [task['initial_cds']],
            objective,
            NUM_INITIAL,
            algorithm=algo,
            seed=SEED,
            diversity=0,
            is_CDS_optimized=True,
            **stop_kwargs,
        )
    else:
        initial = random_seed_sequences(length=length, count=NUM_INITIAL, seed=SEED)
        result = optimize(
            scorer, length,
            initial,
            objective,
            NUM_INITIAL,
            algorithm=algo,
            seed=SEED,
            diversity=0,
            **stop_kwargs,
        )

    # SaturationConfig always returns a SaturationResult (or a list for CDS).
    if isinstance(result, list):
        # is_CDS_optimized with multiple CDSs; we have just one
        result = result[0]
    if isinstance(result, SaturationResult):
        result = result[0.99]

    # Top NUM_RETURN candidates (already sorted best-first by _finalize)
    frame = result.to_frame().head(NUM_RETURN)
    return {
        'task_id':      task['id'],
        'algorithm':    algo,
        'sequences':    frame['sequence'].tolist(),
        'scores':       frame['score'].tolist(),
        'query_count':  result.query_count,
        'stop_reason':  result.metadata.get('stop_reason', 'unknown'),
        'trajectory':   result.trajectory,
        'activities':   result.activities.tolist() if result.activities is not None else None,
        'task_names':   list(result.task_names),
        'status':       'ok',
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_task.py <task_index>")
        sys.exit(1)

    task_idx = int(sys.argv[1])
    from tasks import TASKS
    task = TASKS[task_idx]
    task_id = task['id']

    print(f"\n{'='*60}")
    print(f"Task {task_idx}: {task_id}")
    print(f"{'='*60}")

    for algo in ALGORITHMS:
        out_path = result_path(task_id, algo)
        if os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                prev = pickle.load(f)
            print(f"  [skip] {algo} — already done ({prev.get('status','?')})")
            continue

        if algo in GRADIENT_ALGORITHMS:
            print(f"  [skip] {algo} — requires score_soft (not available on bioseq_models scorers)")
            record = {
                'task_id': task_id, 'algorithm': algo,
                'status': 'skipped', 'reason': 'requires_score_soft',
            }
            with open(out_path, 'wb') as f:
                pickle.dump(record, f)
            continue

        print(f"  Running {algo} ...", end=' ', flush=True)
        t0 = time.time()
        try:
            record = run_one(task, algo)
            record['elapsed_s'] = round(time.time() - t0, 1)
            top = max(record['scores'])
            print(f"done in {record['elapsed_s']}s  "
                  f"top={top:.4f}  queries={record['query_count']}  "
                  f"stop={record['stop_reason']}")
        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            tb = traceback.format_exc()
            print(f"FAILED ({elapsed}s): {exc}")
            record = {
                'task_id': task_id, 'algorithm': algo,
                'status': 'error', 'error': str(exc),
                'traceback': tb, 'elapsed_s': elapsed,
            }

        with open(out_path, 'wb') as f:
            pickle.dump(record, f)

    print(f"\nTask {task_idx} complete. Results in {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
