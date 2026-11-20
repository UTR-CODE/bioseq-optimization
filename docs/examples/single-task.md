# Optimus 5-Prime for 5′ UTR design

[Notebook](../../examples/single-task.ipynb) · [Installation](../installation.md)

Maximize predicted mean ribosome load for a 50-nt 5′ UTR. Exclude homopolymers of six or more bases.

## 1. Load the model

```python
import re
from pathlib import Path

import torch
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, random_seed_sequences, select_candidates

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = load_model("optimus5prime", device=device)

run = Path("outputs/optimus_5utr")
```

## 2. Generate candidates

```python
def no_homopolymer6(sequences):
    return [re.search(r"([ACGT])\1{5}", sequence) is None for sequence in sequences]


seeds = random_seed_sequences(50, 100, seed=42)

designs = generate_designs(
    model, length=50, initial=seeds, objective="maximize",
    constraint=no_homopolymer6,
    output_dir=run / "designs",
)
designs.groupby("algorithm")["score"].agg(["size", "median", "max"])
```

## 3. Discover motifs

```python
motifs = call_motifs(designs, model, reverse_complement=False, output_dir=run / "motifs")
motifs.support.query("retained")[["motif_id", "seqlets", "parents", "sites_on_unique_DNA"]]
```

## 4. Select a panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.2))
selected = select_candidates(designs, motifs=motifs, top_n=10, min_score=gate, output_dir=run / "selected")
selected[["selection_rank", "sequence", "score", "present_motifs"]]
```
