# Single-task sequence design

[Home](../../README.md) · [Core API](../core-api.md) · [Notebook](../../examples/single-task.ipynb)

Use a trained PyTorch `model` to predict one activity from `[batch,4,200]` sequences. The design objective is activity maximization.

## 1. Connect the predictor

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_optimization import generate_designs, call_motifs, select_candidates
from sequence_optimization import SequencePredictor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
scorer = SequencePredictor(model, tasks=["activity"], device=DEVICE)
objective = ["maximize"]
```

Use `["minimize"]` for low-activity design. A standard channels-first PyTorch module can also be supplied directly to `generate_designs` and `call_motifs`.

## 2. Generate candidate designs

```python
designs = generate_designs(
    scorer, length=200, objective=objective,
    num_candidates=100, seed=42, output_dir="outputs/single_task",
)
timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
```

The 11 optimizers share a starting population, run to a plateau, and each contribute 100 designs at the 90% observed-gain stage. The returned table contains sequences, individual activity predictions, scores, and algorithm provenance.

## 3. Call motifs from the designs

```python
motifs = call_motifs(
    designs, model=scorer, output_dir="outputs/single_task_motifs",
)
print(motifs.support)
```

This calculates contributions for the optimization objective, extracts positive seqlets, and discovers supported PWMs. The catalog is saved as `catalog.meme`.

## 4. Select a motif-diverse panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.9))
selected = select_candidates(
    designs, top_n=10, motifs=motifs, min_score=gate,
    output_dir="outputs/single_task_top10",
)
print(selected[["selection_rank", "sequence_id", "score", "present_motifs", "algorithms"]])
```

The selector scans the catalog onto the designs and prioritizes motif composition, matched motif positions, and prediction score. CSV, FASTA, motif sites, distance matrices, and the selection trace are exported together.

## 5. Hamming selection without motifs

For a sequence-distance panel, omit the `motifs` argument. This mode also works directly after design generation, independently of motif discovery.

```python
hamming_panel = select_candidates(
    designs, top_n=10, min_score=gate,
    output_dir="outputs/single_task_hamming",
)
print(hamming_panel.attrs["selection_method"])  # hamming
```

To use one optimization method, set `algorithms=["fast_seqprop"]` in `generate_designs`. The remaining two core functions stay the same.
