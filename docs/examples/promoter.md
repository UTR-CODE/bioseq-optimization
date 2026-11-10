# Promoter design

[Home](../../README.md) · [Notebook](../../examples/promoter.ipynb)

Optimize an 80-nt promoter insert for high predicted activity. This example uses a predictor trained on a 110-nt construct with a 17-nt left flank and a 13-nt right flank.

Input: a trained PyTorch `model` mapping `[batch,4,110]` constructs to a single activity output.

## 1. Define the optimization region and fixed flanks

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_optimization import SequencePredictor, onehot, random_seed_sequences
from sequence_optimization.workflows import generate_candidates

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEFT = "TGCATTTTTTTCACATC"
RIGHT = "GGTTACGGCTGTT"
left = torch.tensor(onehot([LEFT]))
right = torch.tensor(onehot([RIGHT]))

def promoter_input(p):
    construct = torch.cat([
        left.to(p).expand(len(p), -1, -1),
        p,
        right.to(p).expand(len(p), -1, -1),
    ], dim=1)
    return construct.transpose(1, 2)

scorer = SequencePredictor(
    model, tasks=["promoter_activity"], device=DEVICE,
    batch_size=512, input_transform=promoter_input,
)
```

The optimization variables cover the 80-nt insert. Fixed flanks are concatenated during both discrete and gradient scoring. For a predictor trained directly on the insert, use `SequencePredictor(model, tasks=["promoter_activity"], device=DEVICE)`.

## 2. Generate candidates and record runtime

```python
initial = random_seed_sequences(80, 100, seed=42)
candidates, timing = generate_candidates(
    scorer, 80, initial, ["maximize"],
    output_dir="outputs/promoter", num_candidates=100, seed=42,
)
print(timing[["algorithm", "seconds", "queries", "score_median"]])
```

## 3. Select a high-activity sequence-diverse panel

Retain the upper activity decile and apply Hamming Max-Min selection. The first sequence has the highest activity; subsequent sequences maximize their minimum distance to the selected panel.

```python
from sequence_optimization.workflows import deduplicate_candidates, select_sequence_panel
unique, provenance = deduplicate_candidates(candidates)
gate = float(unique.score.quantile(0.9))
selected, distances = select_sequence_panel(candidates, top_n=10, min_score=gate)
print(selected[["selection_rank", "sequence_id", "score", "algorithms"]])
```

A promoter motif catalog can replace this sequence-distance stage with the [motif composition and positional selection](../motif-diversity.md).

## 4. Export inserts and complete constructs

```python
from sequence_optimization.workflows import write_fasta
panel = Path("outputs/promoter_top10")
panel.mkdir(parents=True, exist_ok=True)
selected["construct"] = LEFT + selected.sequence + RIGHT
selected.to_csv(panel / "selected_sequences.csv", index=False)
write_fasta(panel / "inserts.fa", selected.sequence_id, selected.sequence)
write_fasta(panel / "constructs.fa", selected.sequence_id, selected.construct)
np.savetxt(panel / "hamming_distances.csv", distances, delimiter=",", fmt="%d")
```

## 5. Calculate contributions and plot a sequence logo

```python
from sequence_optimization import ism_contributions
import matplotlib.pyplot as plt
import logomaker

contribution = ism_contributions(
    selected.sequence.to_numpy(),
    lambda sequences: scorer.score(sequences).scores[:, 0],
    batch_size=512,
)
np.savez_compressed(panel / "ism.npz", observed=contribution.observed, logo=contribution.logo)
fig, ax = plt.subplots(figsize=(14, 2.5))
logomaker.Logo(pd.DataFrame(contribution.logo[0], columns=list("ACGT")), ax=ax)
ax.set_xlabel("Position in the 80-nt insert")
ax.set_ylabel("ISM contribution")
fig.tight_layout()
fig.savefig(panel / "sequence01_ism.png", dpi=180)
```

ISM evaluates three alternative nucleotides per position. The logo displays the contribution of the observed base relative to the four-base mean prediction.
