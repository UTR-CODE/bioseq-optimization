# Optimus 5-Prime for 5′ UTR design

[Notebook](../../examples/single-task.ipynb) · [Installation](../installation.md)

Maximize the predicted mean ribosome load of a 50-nt 5′ UTR. Start with 100 random sequences. MotifSift replaces seeds that contain homopolymers of six bases or longer. Install MEME Suite with the notebook dependencies. The first model load downloads the pretrained checkpoint.

## 1. Load the model

```python
import os
import re
from pathlib import Path

import torch
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, random_seed_sequences, select_candidates

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("optimus5prime", device=device)

run = Path("outputs/optimus_5utr")
os.makedirs(run, exist_ok=True)
```

## 2. Generate candidates

Run the eleven optimizers and keep 100 candidates per optimizer. The saved tables show the previous full run, which used a prefiltered seed pool. New runs use the current MotifSift defaults.

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

Extract positive seqlets from in silico mutagenesis scores and discover motifs with XSTREME.

```python
motifs = call_motifs(
    designs, model=model, reverse_complement=False,
    output_dir=run / "motifs",
)
motifs.support.query("retained")[["motif_id", "seqlets", "parents", "sites_on_unique_DNA"]]
```

## 4. Select a panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.2))
selected = select_candidates(designs, motifs=motifs, top_n=10, min_score=gate, output_dir=run / "selected")
selected[["selection_rank", "sequence", "score", "present_motifs"]]
```
