# MPRALegNet for cell-selective enhancer design

[Notebook](../../examples/multiple-models.ipynb) · [Installation](../installation.md)

Favor K562 activity over HepG2 and WTC11 using the ten-fold MPRALegNet ensemble for each cell type.

## 1. Load the model

```python
from pathlib import Path

import torch
from bioseq_models import load_model
from MotifSift import CombinedScorer, MultiTaskObjective, call_motifs, generate_designs, random_seed_sequences, select_candidates

device = "cuda" if torch.cuda.is_available() else "cpu"
cells = ["K562", "HepG2", "WTC11"]
model = CombinedScorer({
    cell: load_model(f"agarwal_{cell.lower()}", device=device)
    for cell in cells
})
objective = MultiTaskObjective(high=["K562"], low=["HepG2", "WTC11"])
seeds = random_seed_sequences(230, 100, seed=42)
run = Path("outputs/mpralegnet")
```

## 2. Generate candidates

The saved outputs show an earlier full run.

```python
designs = generate_designs(
    model, length=230, initial=seeds, objective=objective,
    output_dir=run / "designs",
)
designs.groupby("algorithm")["score"].agg(["size", "median", "max"])
```

## 3. Discover motifs

```python
motifs = call_motifs(designs, model, output_dir=run / "motifs")
motifs.support.query("retained")[["motif_id", "seqlets", "parents", "sites_on_unique_DNA"]]
```

## 4. Select a panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.2))
selected = select_candidates(designs, motifs=motifs, top_n=10, min_score=gate, output_dir=run / "selected")
selected[["selection_rank", "sequence", "score", "present_motifs"]]
```
