# Multi-task design: one output high, the others low

[Home](../../README.md) · [Core API](../core-api.md) · [Notebook](../../examples/multi-task.ipynb)

The trained PyTorch `model` returns three activities: `target`, `off1`, and `off2`.

## 1. Define the target and off-target objective

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_optimization import generate_designs, call_motifs, select_candidates
from sequence_optimization import SequencePredictor, MultiTaskObjective
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
scorer = SequencePredictor(model, tasks=["target", "off1", "off2"], device=DEVICE)
objective = MultiTaskObjective(
    high=["target"], low=["off1", "off2"], aggregation="mean",
)
```

With equal scales, the score is `(target - off1 - off2) / 3`. Add further off-target names to `low` for models with additional outputs.

## 2. Generate candidate designs

```python
designs = generate_designs(
    scorer, length=200, objective=objective,
    num_candidates=100, seed=42, output_dir="outputs/multi_task",
)
timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
```

The 11 optimizers share a starting population, run to a plateau, and each contribute 100 designs at the 90% observed-gain stage. The returned table contains sequences, individual activity predictions, scores, and algorithm provenance.

## 3. Call motifs from the designs

```python
motifs = call_motifs(
    designs, model=scorer, output_dir="outputs/multi_task_motifs",
)
print(motifs.support)
```

This calculates contributions for the optimization objective, extracts positive seqlets, and discovers supported PWMs. The catalog is saved as `catalog.meme`.

## 4. Select a motif-diverse panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.9))
selected = select_candidates(
    designs, top_n=10, motifs=motifs, min_score=gate,
    output_dir="outputs/multi_task_top10",
)
print(selected[["selection_rank", "activity_target", "activity_off1", "activity_off2", "score", "present_motifs"]])
```

The selector scans the catalog onto the designs and prioritizes motif composition, matched motif positions, and prediction score. CSV, FASTA, motif sites, distance matrices, and the selection trace are exported together.

## 5. Hamming selection without motifs

For a sequence-distance panel, omit the `motifs` argument. This mode also works directly after design generation, independently of motif discovery.

```python
hamming_panel = select_candidates(
    designs, top_n=10, min_score=gate,
    output_dir="outputs/multi_task_hamming",
)
print(hamming_panel.attrs["selection_method"])  # hamming
```

## 6. Threshold and specificity objectives

A worst-margin objective can target probability thresholds of 0.8 for the target and 0.2 for each off-target:

```python
bounded_objective = MultiTaskObjective(
    high=["target"], low=["off1", "off2"],
    lower_bounds={"target": 0.8},
    upper_bounds={"off1": 0.2, "off2": 0.2},
    aggregation="worst",
)
```

Its score is `min(target - 0.8, 0.2 - off1, 0.2 - off2)`. Pass it as `objective` during generation; motif calling then uses the same criterion.

For direct specificity `target - max(off1, off2)`, construct pairwise activity contrasts:

```python
def activity_and_contrasts(y):
    return torch.cat([y, y[:, :1] - y[:, 1:]], dim=1)

specificity_scorer = SequencePredictor(
    model, device=DEVICE,
    tasks=["target", "off1", "off2", "margin1", "margin2"],
    output_transform=activity_and_contrasts,
)
specificity_objective = MultiTaskObjective(high=["margin1", "margin2"], aggregation="worst")
```

Use `specificity_scorer` and `specificity_objective` for generation and the same scorer for motif calling. The [enhancer example](enhancer.md) uses this objective.
