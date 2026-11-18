# Evaluation

[Home](../README.md)

Evaluate the optimized pool and selected panel using prediction quality, sequence diversity, motif diversity, and computational cost.

## Prediction quality

Report activity and specificity distributions per algorithm. For a target-specific task, retain all target and off-target outputs together with the scalar specificity score.

```python
import pandas as pd
timing = pd.DataFrame(designs.attrs["timing"])
summary = designs.groupby("algorithm").score.agg(["min", "median", "max"])
print(summary.join(timing.set_index("algorithm")[["seconds", "queries"]]))
```

## Sequence and motif diversity

For each selected panel, report:

- Number of unique sequences and contributing optimization algorithms.
- Minimum and mean pairwise Hamming distance.
- Motif coverage and number of distinct presence patterns.
- Minimum and mean Soft-Jaccard composition distance.
- Matched positional distances and motif-match counts.

```python
import numpy as np
matrices = np.load("outputs/enhancer_top10/distance_matrices.npz")
lookup = {sid: i for i, sid in enumerate(matrices["sequence_ids"])}
idx = np.array([lookup[sid] for sid in selected.sequence_id])
triangle = np.triu_indices(len(idx), 1)
for name in ["composition", "position_bp"]:
    values = matrices[name][np.ix_(idx, idx)][triangle]
    if name == "position_bp":
        matched = matrices["matched_counts"][np.ix_(idx, idx)][triangle]
        values = values[matched > 0]
    print(name, {"minimum": float(values.min()) if len(values) else None,
                 "mean": float(values.mean()) if len(values) else None})
```

Interpret positional distances together with `matched_counts`. Report positional summaries over pairs with compatible motif matches.

## Computational cost

Record total wall time, per-algorithm runtime, CPU time, proposal count, and actual model-scoring rows. For shared-GPU runs, also record queue time and batch utilization.

## Current result inputs

The current K562 and HepG2 catalogs and candidate pools are in `../../results/`. Their downstream panels have not yet been recomputed. The workspace manifest records the source and SHA-256 of each included file. No tests were run while assembling this workspace.
