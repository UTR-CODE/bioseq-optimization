# 5′UTR design

[Home](../../README.md) · [Notebook](../../examples/5utr.ipynb)

Optimize 50-nt 5′UTRs for high predicted translation. This example uses mean ribosome load (MRL) as the activity output.

Input: a trained PyTorch `model` mapping `[batch,4,50]` sequences to predicted MRL.

## 1. Connect the predictor

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_optimization import SequencePredictor, random_seed_sequences
from sequence_optimization.workflows import generate_candidates

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
scorer = SequencePredictor(model, tasks=["mrl"], device=DEVICE, batch_size=512)
initial = random_seed_sequences(50, 100, seed=42)
```

## 2. Generate 1,100 candidates

```python
candidates, timing = generate_candidates(
    scorer, 50, initial, ["maximize"],
    output_dir="outputs/utr", num_candidates=100, seed=42,
)
print(timing[["algorithm", "seconds", "queries", "score_median"]])
```

## 3. Select ten diverse high-MRL sequences

```python
from sequence_optimization.workflows import deduplicate_candidates, select_sequence_panel
unique, provenance = deduplicate_candidates(candidates)
gate = float(unique.score.quantile(0.9))
selected, distances = select_sequence_panel(candidates, top_n=10, min_score=gate)
print(selected[["selection_rank", "sequence_id", "score", "algorithms"]])
print("Minimum pairwise Hamming distance:", distances[np.triu_indices(10, 1)].min())
```

This stage uses sequence diversity. A task-specific catalog of contribution-derived sequence patterns can also be supplied to the [motif-diversity workflow](../motif-diversity.md).

## 4. Annotate upstream AUGs and export

```python
from sequence_optimization.workflows import write_fasta
panel = Path("outputs/utr_top10")
panel.mkdir(parents=True, exist_ok=True)
selected["upstream_ATG_positions0"] = selected.sequence.map(
    lambda s: ";".join(str(i) for i in range(len(s)-2) if s[i:i+3] == "ATG")
)
selected["has_upstream_ATG"] = selected.upstream_ATG_positions0.ne("")
selected.to_csv(panel / "selected_sequences.csv", index=False)
write_fasta(panel / "selected_sequences.fa", selected.sequence_id, selected.sequence)
np.savetxt(panel / "hamming_distances.csv", distances, delimiter=",", fmt="%d")
```

## 5. Calculate ISM contributions and plot a logo

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
fig, ax = plt.subplots(figsize=(12, 2.5))
logomaker.Logo(pd.DataFrame(contribution.logo[0], columns=list("ACGT")), ax=ax)
ax.set_xlabel("Position in the 50-nt UTR")
ax.set_ylabel("ISM contribution")
fig.tight_layout()
fig.savefig(panel / "sequence01_ism.png", dpi=180)
```
