"""Paired Optimus experiment: only the retrospective selection rule changes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from bioseq_models import load_model
from sequence_optimization import (
    SaturationConfig,
    available_algorithms,
    random_seed_sequences,
)
from sequence_optimization.algorithms import optimize
from check_ga_parents import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--relaxed-only", action="store_true")
args = parser.parse_args()

out = Path('results') / time.strftime('optimus_relaxed_knee_%Y%m%d_%H%M%S')
out.mkdir(parents=True)
print(out.resolve(), flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model('optimus5prime', device=device, batch_size=256)
initial = random_seed_sequences(length=50, count=100, seed=42)
rows = []
for algorithm in [name for name in available_algorithms() if name != "random"]:
    for relaxed in ([True] if args.relaxed_only else [False, True]):
        start = time.monotonic()
        result = optimize(model, 50, initial, ['maximize'], 100, algorithm=algorithm,
                          seed=42, saturation=SaturationConfig(relaxed_knee=relaxed))
        frame = result.selected.to_frame()
        meta = result.selected.metadata
        assert len(frame) == frame.sequence.nunique() == 100
        if relaxed:
            gain = meta['observed_peak_gain']
            baseline = meta['baseline_population_mean']
            assert result.knee.scores.mean() >= baseline + .85 * gain - 1e-9
            assert frame.score.mean() >= baseline + .8 * gain - 1e-9
            assert meta['selection_mean_floor_satisfied']
        codes = np.array([list(s.encode()) for s in frame.sequence])
        labels = fcluster(linkage(pdist(codes, 'hamming') * 50, method='complete'), 10, criterion='distance')
        sizes = pd.Series(labels).value_counts()
        row = dict(algorithm=algorithm, relaxed=relaxed, seconds=time.monotonic()-start,
                   queries=result.total_query_count, **statistics(frame),
                   knee_mean=float(result.knee.scores.mean()),
                   gain_mean=meta['selected_normalized_gain_mean'],
                   gain_min=meta['selected_normalized_gain_minimum'],
                   largest_cluster=int(sizes.max()), top_two_share=float(sizes.iloc[:2].sum()/100))
        rows.append(row)
        prefix = f'{algorithm}_{"relaxed" if relaxed else "original"}'
        frame.to_csv(out / f'{prefix}_candidates.csv', index=False)
        (out / f'{prefix}_metadata.json').write_text(json.dumps(meta, indent=2))
        pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
        print(row, flush=True)
