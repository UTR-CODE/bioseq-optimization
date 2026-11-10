# Single-task sequence design

[Home](../../README.md) · [Notebook](../../examples/single-task.ipynb) · [Multi-task](multi-task.md) · [Multiple models](multiple-models.md)

Optimize sequences using one predicted activity. The inputs are a trained PyTorch `model`, the sequence length, and a starting population. The model accepts `[batch,4,length]` and returns `[batch]` or `[batch,1]`.

## 1. Connect the model and generate starting sequences

```python
from pathlib import Path
import numpy as np
import torch
from sequence_optimization import SequencePredictor, random_seed_sequences

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LENGTH = 200
scorer = SequencePredictor(model, tasks=["activity"], device=DEVICE, batch_size=512)
initial = random_seed_sequences(LENGTH, 100, seed=42)
objective = ["maximize"]
print(scorer.score(initial[:4]).scores)
```

The fitness is the predicted activity, `S(x) = a(x)`. Set `objective = ["minimize"]` to design low-activity sequences. Input layouts and fixed contexts are handled with [input transforms](../pytorch-models.md).

## 2. Optimize with one algorithm

```python
from sequence_optimization import optimize, SaturationConfig
saturation = SaturationConfig(
    fractions=(0.9,), selection="fraction", maximum_queries=None,
)
result = optimize(
    scorer, LENGTH, initial, objective, 100,
    algorithm="fast_seqprop", saturation=saturation,
    diversity=5, seed=42,
)
retained = result.at(0.9)
print(result.stop_reason, result.total_query_count)
print(retained.to_frame().head())
```

This run retains 100 candidates at the 90% observed-gain stage. Change `algorithm` to any method in [the algorithm table](../algorithms.md).

## 3. Generate candidates with all 11 algorithms

```python
from sequence_optimization.workflows import generate_candidates
candidates, timing = generate_candidates(
    scorer, LENGTH, initial, objective,
    output_dir="outputs/single_task", num_candidates=100,
    saturation=saturation, diversity=5, seed=42,
)
print(candidates.groupby("algorithm").size())
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
```

The combined table contains 1,100 records with `sequence`, `algorithm`, `score`, and `activity_activity` columns, together with record and sequence identifiers.

## 4. Select and export a high-activity panel

Retain the upper activity decile, then select ten sequences by Hamming Max-Min diversity.

```python
from sequence_optimization.workflows import deduplicate_candidates, select_sequence_panel, write_fasta
unique, provenance = deduplicate_candidates(candidates)
gate = float(unique.score.quantile(0.9))
selected, distances = select_sequence_panel(candidates, top_n=10, min_score=gate)

panel = Path("outputs/single_task_top10")
panel.mkdir(parents=True, exist_ok=True)
selected.to_csv(panel / "selected_sequences.csv", index=False)
write_fasta(panel / "selected_sequences.fa", selected.sequence_id, selected.sequence)
np.savetxt(panel / "hamming_distances.csv", distances, delimiter=",", fmt="%d")
print(selected[["selection_rank", "sequence_id", "activity_activity", "score", "algorithms"]])
```

For motif-based panel selection, use the same candidate pool with [motif discovery](../motif-discovery.md) and [motif diversity](../motif-diversity.md). Single-task ISM contributions use `lambda sequences: scorer.score(sequences).scores[:, 0]` as the fitness function.
