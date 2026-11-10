"""Eleven optimizer candidate pools with explicit plateau and provenance records."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import logging
import multiprocessing as mp
import sys
import time
import traceback
import numpy as np
import pandas as pd
from .algorithms import optimize
from .api import as_score_batch
from .saturation import SaturationConfig
from .seeds import random_seed_sequences

DESIGN_ALGORITHMS = (
    'adalead', 'basesearch', 'beam_search', 'cross_entropy_method', 'den',
    'fast_seqprop', 'genetic_algorithm', 'ism_greedy', 'ledidi',
    'simulated_annealing', 'vanilla_activation_maximization',
)


def write_fasta(path, ids, sequences):
    ids, sequences = list(ids), list(sequences)
    if len(ids) != len(sequences) or len(set(map(str, ids))) != len(ids):
        raise ValueError('FASTA IDs must be unique and match the sequence count')
    if any(any(c.isspace() for c in str(i)) for i in ids):
        raise ValueError('FASTA IDs cannot contain whitespace')
    Path(path).write_text(''.join(f'>{i}\n{s}\n' for i, s in zip(ids, sequences)))


def deduplicate_candidates(records):
    """Collapse exact DNA, preserving a separate many-to-one provenance table."""
    records = records.copy()
    records['sequence_id'] = records.sequence.map(
        lambda s: 'S_'+hashlib.sha256(s.encode()).hexdigest()[:20])
    if records.groupby('sequence_id').sequence.nunique().max() != 1:
        raise RuntimeError('Sequence ID collision')
    unique = records.sort_values(['score', 'record_id'], ascending=[False, True]).drop_duplicates('sequence').copy()
    sources = records.groupby('sequence_id').algorithm.agg(lambda v: ';'.join(sorted(set(v))))
    unique['algorithms'] = unique.sequence_id.map(sources)
    return unique.reset_index(drop=True), records


def _one_algorithm(scorer, initial, objective, name, count, seed, diversity,
                   saturation, folder, options):
    folder.mkdir(parents=True, exist_ok=True)
    began, cpu = time.perf_counter(), time.process_time()
    result = optimize(scorer, len(initial[0]), initial, objective, count,
                      algorithm=name, seed=seed, diversity=diversity,
                      saturation=saturation, count_repeated=True, **options)
    if result.stop_reason != 'plateau':
        raise RuntimeError(f'{name}: expected plateau, got {result.stop_reason}')
    selected = result.at(.9)
    seconds, cpu_seconds = time.perf_counter()-began, time.process_time()-cpu
    if len(selected.sequences) != count or len(set(selected.sequences)) != count:
        raise RuntimeError(f'{name}: expected {count} distinct candidates')
    fresh = as_score_batch(scorer.score(selected.sequences)).scores
    np.testing.assert_allclose(fresh, selected.activities, atol=1e-3, rtol=1e-4)
    d = pd.DataFrame({'record_id':[f'{name}_{i+1:03}' for i in range(count)],
                      'sequence':selected.sequences, 'algorithm':name, 'score':selected.scores})
    for j, task in enumerate(scorer.task_names):
        d['activity_'+task] = fresh[:, j]
    d.to_csv(folder/'candidates.csv', index=False)
    write_fasta(folder/'candidates.fa', d.record_id, d.sequence)
    (folder/'trajectory.json').write_text(json.dumps(result.trajectory, indent=2))
    info = dict(algorithm=name, seconds=seconds, cpu_seconds=cpu_seconds,
                queries=result.total_query_count, stop_reason=result.stop_reason,
                candidates=count, score_min=float(d.score.min()),
                score_median=float(d.score.median()), score_max=float(d.score.max()))
    (folder/'run.json').write_text(json.dumps(info, indent=2)+'\n')
    return d, info


def _configuration(initial, count, names, seed, diversity, saturation, options):
    if not names or len(names) != len(set(names)) or set(names)-set(DESIGN_ALGORITHMS):
        raise ValueError('Choose distinct non-random DESIGN_ALGORITHMS')
    if len(initial) < count or not len(initial) or len(set(map(len, initial))) != 1:
        raise ValueError('Provide at least num_candidates equal-length initial sequences')
    if any(set(s)-set('ACGT') for s in initial):
        raise ValueError('Initial sequences must contain only A/C/G/T')
    if .9 not in saturation.fractions or saturation.selection != 'fraction' or saturation.maximum_queries is not None:
        raise ValueError('This workflow requires 90% gain fraction selection with no query ceiling')
    forbidden = {'budget','saturation','algorithm','seed','diversity','count_repeated','num_candidates'} & set(options)
    if forbidden:
        raise ValueError(f'Use named workflow arguments instead of algorithm_options: {forbidden}')
    return dict(algorithms=list(names), candidates_per_algorithm=count,
                initial_count=len(initial), sequence_length=len(initial[0]), seed=seed,
                minimum_hamming=diversity, saturation=asdict(saturation),
                algorithm_options=options, candidate_rows=count*len(names))


def _prepare(out, initial, config):
    # Refuse overwriting results from another search.
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'Use a new empty output directory: {out}')
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    pd.DataFrame({'sequence':initial}).to_csv(out/'initial_sequences.csv', index=False)


def _combine(out, names, count):
    records = pd.concat([pd.read_csv(out/n/'candidates.csv') for n in names], ignore_index=True)
    if len(records) != len(names)*count:
        raise RuntimeError('Incomplete candidate pool')
    unique, provenance = deduplicate_candidates(records)
    provenance.to_csv(out/'all_candidates.csv', index=False)
    unique.to_csv(out/'unique_candidates.csv', index=False)
    write_fasta(out/'unique_candidates.fa', unique.sequence_id, unique.sequence)
    runtime = pd.DataFrame([json.loads((out/n/'run.json').read_text()) for n in names])
    runtime.to_csv(out/'algorithm_summary.csv', index=False)
    return provenance, runtime


def generate_candidates(scorer, length, initial, objective, *, output_dir,
                        num_candidates=100, algorithms=DESIGN_ALGORITHMS, seed=42,
                        diversity=5, saturation=None, algorithm_options=None):
    """Readable serial reference: N algorithms x candidates; random excluded."""
    initial = np.asarray(initial, dtype=str)
    if any(len(s) != length for s in initial):
        raise ValueError('Initial sequence length mismatch')
    names, options = tuple(algorithms), dict(algorithm_options or {})
    sat = saturation or SaturationConfig(fractions=(.9,), selection='fraction', maximum_queries=None)
    config = _configuration(initial, num_candidates, names, seed, diversity, sat, options)
    config["objective"] = repr(objective)
    config["task_names"] = list(scorer.task_names)
    out = Path(output_dir); _prepare(out, initial, config)
    started = time.perf_counter()
    for name in names:
        _one_algorithm(scorer, initial, objective, name, num_candidates, seed, diversity, sat, out/name, options)
    records, runtime = _combine(out, names, num_candidates)
    (out/'walltime.json').write_text(json.dumps({'seconds':time.perf_counter()-started,'mode':'serial'}))
    return records, runtime


def _parallel_worker(client, initial, objective, name, count, seed, diversity, sat, folder, options):
    with (folder.parent/f'{name}.log').open('w', buffering=1) as log:
        sys.stdout = sys.stderr = log
        import torch
        import numba
        torch.set_num_threads(1); numba.set_num_threads(1)
        logging.getLogger('sequence_optimization.algorithms').addHandler(logging.StreamHandler(log))
        try:
            _one_algorithm(client, initial, objective, name, count, seed, diversity, sat, folder, options)
        except BaseException:
            traceback.print_exc()
            raise


def generate_candidates_parallel(scorer_factory, length, initial, objective, *,
                                 task_names, output_dir, num_candidates=100,
                                 algorithms=DESIGN_ALGORITHMS, seed=42, diversity=5,
                                 max_batch_size=512, max_soft_batch_size=64,
                                 batch_wait_ms=5., start_method='spawn', saturation=None,
                                 algorithm_options=None):
    """Independent CPU optimizers; one model-owning batched inference service.

    For spawn use a module-level factory and an if __name__ == '__main__' guard.
    Instantiate the model ONLY inside scorer_factory, never in the parent.
    """
    from .parallel import ScoringClient, scoring_service
    initial = np.asarray(initial, dtype=str)
    if any(len(s) != length for s in initial):
        raise ValueError('Initial sequence length mismatch')
    names, options = tuple(algorithms), dict(algorithm_options or {})
    sat = saturation or SaturationConfig(fractions=(.9,), selection='fraction', maximum_queries=None)
    config = _configuration(initial, num_candidates, names, seed, diversity, sat, options)
    config["objective"] = repr(objective)
    config["task_names"] = list(task_names)
    config["scorer_factory"] = scorer_factory.__module__+"."+scorer_factory.__qualname__
    out = Path(output_dir); _prepare(out, initial, config)
    ctx = mp.get_context(start_method)
    requests, ready = ctx.Queue(maxsize=2*len(names)), ctx.Queue()
    replies, stopped = [ctx.Queue(maxsize=2) for _ in names], ctx.Event()
    service = ctx.Process(target=scoring_service, args=(scorer_factory,requests,replies,stopped,ready),
        kwargs=dict(max_batch_size=max_batch_size,max_soft_batch_size=max_soft_batch_size,
                    batch_wait_ms=batch_wait_ms,metrics_path=str(out/'gpu_batches.csv')))
    workers=[]; started=time.perf_counter()
    try:
        service.start()
        # No arbitrary model-download timeout; detect a dead process while waiting.
        import queue
        while True:
            try: ok, error = ready.get(timeout=1.); break
            except queue.Empty:
                if not service.is_alive(): raise RuntimeError('Scoring service exited during initialization')
        if not ok: raise RuntimeError(error)
        for i,name in enumerate(names):
            client=ScoringClient(i,requests,replies[i],stopped,task_names=task_names,
                                 batch_size=max_batch_size,soft_batch_size=max_soft_batch_size)
            worker=ctx.Process(target=_parallel_worker,args=(client,initial,objective,name,num_candidates,
                seed,diversity,sat,out/name,options))
            worker.start(); workers.append(worker)
        while any(p.is_alive() for p in workers):
            if not service.is_alive(): raise RuntimeError('Scoring service failed; inspect GPU log/metrics')
            for p in workers:
                if p.exitcode not in (None,0): raise RuntimeError(f'Optimizer failed: {p.name}; inspect algorithm logs')
            time.sleep(.1)
        if any(p.exitcode != 0 for p in workers): raise RuntimeError('An optimizer failed')
        requests.put(None); service.join(timeout=30)
        if service.is_alive() or service.exitcode != 0: raise RuntimeError('Scoring service did not finish cleanly')
        records,runtime=_combine(out,names,num_candidates)
        (out/'walltime.json').write_text(json.dumps({'seconds':time.perf_counter()-started,'mode':'parallel',
            'timing_note':'Per-algorithm seconds include queue wait and contention; not isolated speed benchmarks.'}))
        return records,runtime
    finally:
        stopped.set()
        for p in [*workers,service]:
            if p.is_alive():p.terminate();p.join(timeout=5)
        for q in [requests,ready,*replies]:q.cancel_join_thread();q.close()


def select_sequence_panel(records, *, top_n=10, min_score, distance_tie_nt=1):
    """Fallback for tasks without a validated motif catalog: DNA Hamming Max-Min.

    This selects sequence diversity only and makes no claim of motif diversity.
    """
    unique,_=deduplicate_candidates(records)
    pool=unique[unique.score>=min_score].copy().reset_index(drop=True)
    if isinstance(top_n,bool) or not isinstance(top_n,int) or top_n<1 or distance_tie_nt<0:
        raise ValueError('Invalid panel size or distance tie tolerance')
    if len(pool)<top_n:raise ValueError('Too few unique sequences pass min_score')
    if pool.sequence.str.len().nunique()!=1:raise ValueError('Hamming requires equal lengths')
    codes=np.array([list(s) for s in pool.sequence]);distance=(codes[:,None,:]!=codes[None,:,:]).sum(2)
    chosen=[int(pool.score.to_numpy().argmax())]
    while len(chosen)<top_n:
        remaining=[i for i in range(len(pool)) if i not in chosen]
        nearest=distance[np.ix_(remaining,chosen)].min(1)
        best=nearest.max();stage=[i for i,v in zip(remaining,nearest) if v>=best-distance_tie_nt]
        chosen.append(min(stage,key=lambda i:(-pool.score.iloc[i],pool.sequence_id.iloc[i])))
    selected=pool.iloc[chosen].copy();selected.insert(0,'selection_rank',range(1,top_n+1))
    return selected,distance[np.ix_(chosen,chosen)]
