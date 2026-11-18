# Optimus 5-Prime for 5′ UTR design

[Notebook](../../examples/single-task.ipynb) · [Installation](../installation.md)

Maximize the predicted mean ribosome load of a 50-nt 5′ UTR. Reuse the 100 starting sequences and exclude runs of six identical bases, as in the Extended Data analysis. Install MEME Suite with the notebook dependencies. The first model load downloads the pretrained checkpoint.

## 1. Load the model

```python
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
from MotifSift import call_motifs, select_candidates
from optimus_support import no_homopolymer6
from reproduce import recorded_designs, recorded_model

model = recorded_model("optimus5prime")
seeds = pd.read_csv("data/optimus_initial_sequences.csv")["sequence"].tolist()

run = Path(mkdtemp(prefix="motifsift-example-"))
```

## 2. Generate candidates

Run the eleven optimizers and keep 100 candidates per optimizer. The displayed tables come from the full run reproduced by `examples/reproduce.py`.

```python
designs = recorded_designs(
    "optimus", model, length=50, initial=seeds, objective="maximize",
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
