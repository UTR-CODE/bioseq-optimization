"""Run the updated GA on the Optimus README setup and save a direct comparison."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from bioseq_models import load_model
from sequence_optimization import random_seed_sequences
from sequence_optimization.algorithms import optimize


def statistics(frame):
    codes = np.array([list(s) for s in frame.sequence])
    distances = (codes[:, None] != codes[None, :]).sum(axis=2)
    pair = distances[np.triu_indices(len(frame), 1)]
    np.fill_diagonal(distances, codes.shape[1] + 1)
    nearest = distances.min(axis=1)
    return dict(mean_hamming=float(pair.mean()), minimum_hamming=int(pair.min()),
                nearest_median=float(np.median(nearest)), close_neighbor_fraction=float((nearest <= 10).mean()),
                mrl_mean=float(frame.activity_mrl.mean()), mrl_std=float(frame.activity_mrl.std()),
                mrl_p05=float(frame.activity_mrl.quantile(.05)), mrl_p95=float(frame.activity_mrl.quantile(.95)))


def main():
    out = Path('results') / time.strftime('optimus_ga_parents_%Y%m%d_%H%M%S')
    out.mkdir(parents=True)
    model = load_model('optimus5prime', device='cuda', batch_size=256)
    initial = random_seed_sequences(length=50, count=100, seed=42)
    started = time.monotonic()
    result = optimize(model, 50, initial, ['maximize'], 100,
                      algorithm='genetic_algorithm', seed=42, parent_pool_multiplier=5)
    seconds = time.monotonic() - started
    selected = result.selected if hasattr(result, 'selected') else result
    frame = selected.to_frame()
    assert len(frame) == frame.sequence.nunique() == 100
    frame.to_csv(out / 'candidates.csv', index=False)
    (out / 'metadata.json').write_text(json.dumps(selected.metadata, indent=2))
    old = pd.read_csv('results/examples_profile_20260917_200756/01_optimus_mrl_genetic_algorithm_candidates.csv')
    rows = [dict(version='previous', **statistics(old)), dict(version='5N_parent_pool', seconds=seconds, **statistics(frame))]
    assert rows[1]['minimum_hamming'] >= 5
    pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
    print(out.resolve())
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
