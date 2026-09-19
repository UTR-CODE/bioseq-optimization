"""Validate saved Optimus candidates and export a per-algorithm comparison."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(sys.argv[1])
    timings = pd.read_csv(root / 'algorithm_times.csv')
    rows = []
    for entry in timings.to_dict('records'):
        prefix = f"{entry['example']}_{entry['algorithm']}"
        frame = pd.read_csv(root / f'{prefix}_candidates.csv')
        metadata = json.loads((root / f'{prefix}_metadata.json').read_text())
        sequences = frame.sequence.tolist()
        assert len(sequences) == len(set(sequences)) == 100
        codes = np.array([list(s) for s in sequences])
        distances = np.count_nonzero(codes[:, None, :] != codes[None, :, :], axis=2)
        pairs = distances[np.triu_indices(100, 1)]
        assert pairs.min() >= 5
        np.fill_diagonal(distances, codes.shape[1] + 1)
        nearest = np.sort(distances, axis=1)[:, :10]
        row = dict(algorithm=entry['algorithm'], seconds=entry['elapsed_seconds'],
                   old_min=metadata.get('score_greedy_minimum_pairwise_hamming'),
                   new_min=int(pairs.min()),
                   old_mean=metadata.get('score_greedy_mean_pairwise_hamming'),
                   new_mean=float(pairs.mean()),
                   old_10nn=metadata.get('score_greedy_mean_ten_nearest_hamming'),
                   new_10nn=float(nearest.mean()),
                   target=metadata.get('selection_target_mean'),
                   score_mean=float(frame.score.mean()),
                   gain_min=metadata.get('selected_normalized_gain_minimum'),
                   gain_max=metadata.get('selected_normalized_gain_maximum'),
                   fallback=metadata.get('quality_diversity_fallback_reason'))
        if row['old_10nn'] is not None:
            assert row['new_10nn'] >= row['old_10nn'] - 1e-9
            assert abs(row['new_10nn'] - metadata['mean_ten_nearest_hamming']) < 1e-9
            assert abs(nearest[:, 0].mean() - metadata['mean_nearest_neighbor_hamming']) < 1e-9
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(root / 'diversity_comparison.csv', index=False)
    report = ['# Optimus quality and diversity comparison', '',
              '50 nt; 100 candidates per algorithm; seed 42; minimum Hamming 5.',
              'Quality fractions refer to baseline-to-knee gain, not absolute MRL.',
              'Random is a fixed-budget reference and has no saturation selection.', '',
              '| Algorithm | Seconds | Mean Hamming old → new | Mean 10NN old → new | Gain range |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['old_mean'] is None:
            mean = f"— → {row['new_mean']:.2f}"
            nn = f"— → {row['new_10nn']:.2f}"
            gain = '—'
        else:
            mean = f"{row['old_mean']:.2f} → {row['new_mean']:.2f}"
            nn = f"{row['old_10nn']:.2f} → {row['new_10nn']:.2f}"
            gain = f"{row['gain_min']:.3f}–{row['gain_max']:.3f}"
        report.append(f"| {row['algorithm']} | {row['seconds']:.2f} | {mean} | {nn} | {gain} |")
    report += ['', 'Fallbacks: ' + ', '.join(
        f"{row['algorithm']} ({row['fallback']})" for row in rows if row['fallback'])]
    (root / 'diversity_report.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
