# Promoter design

[Home](../../README.md) · [Core API](../core-api.md) · [Notebook](../../examples/promoter.ipynb)

Optimize an 80-nt promoter insert with a model trained on a 110-nt construct. Fixed flanks are included by the input transform.

## 1. Define the design region

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_optimization import generate_designs, call_motifs, select_candidates
from sequence_optimization import SequencePredictor, onehot
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEFT = "TGCATTTTTTTCACATC"
RIGHT = "GGTTACGGCTGTT"
left = torch.tensor(onehot([LEFT]))
right = torch.tensor(onehot([RIGHT]))

def promoter_input(p):
    return torch.cat([left.to(p).expand(len(p), -1, -1), p,
                      right.to(p).expand(len(p), -1, -1)], dim=1).transpose(1, 2)

scorer = SequencePredictor(model, tasks=["promoter_activity"], device=DEVICE,
                           input_transform=promoter_input)
objective = ["maximize"]
```

For a model trained directly on the insert, use the default input layout.

## 2. Generate candidate designs

```python
designs = generate_designs(
    scorer, length=80, objective=objective,
    num_candidates=100, seed=42, output_dir="outputs/promoter",
)
timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
```

The 11 optimizers run in independent CPU processes with shared batched model scoring, share a starting population, run to a plateau, and each contribute 100 designs at the 90% observed-gain stage. The returned table contains sequences, individual activity predictions, scores, and algorithm provenance.

## 3. Call motifs from the designs

```python
motifs = call_motifs(
    designs, model=scorer, output_dir="outputs/promoter_motifs",
)
print(motifs.support)
```

This calculates contributions for the optimization objective, extracts positive seqlets, and discovers supported PWMs with the built-in backend and held-out parent-group validation. The catalog is saved as `catalog.meme`.

## 4. Select a motif-diverse panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.9))
selected = select_candidates(
    designs, top_n=10, motifs=motifs, min_score=gate,
    output_dir="outputs/promoter_top10",
)
print(selected[["selection_rank", "sequence_id", "score", "present_motifs", "algorithms"]])
```

The selector scans the catalog onto the designs and prioritizes motif composition, matched motif positions, and prediction score. CSV, FASTA, motif sites, distance matrices, and the selection trace are exported together.

## 5. Hamming selection without motifs

For a sequence-distance panel, omit the `motifs` argument. This mode also works directly after design generation, independently of motif discovery.

```python
hamming_panel = select_candidates(
    designs, top_n=10, min_score=gate,
    output_dir="outputs/promoter_hamming",
)
print(hamming_panel.attrs["selection_method"])  # hamming
```

## 6. Export full constructs

```python
from sequence_optimization.workflows import write_fasta
selected["construct"] = LEFT + selected.sequence + RIGHT
selected.to_csv("outputs/promoter_top10/selected_sequences.csv", index=False)
write_fasta("outputs/promoter_top10/constructs.fa", selected.sequence_id, selected.construct)
```

For standalone Python scripts, place model loading and workflow calls under `if __name__ == "__main__":`. Notebook cells call the functions directly. See [Parallel scoring](../parallel-scoring.md) for batch-size controls and utilization metrics.
