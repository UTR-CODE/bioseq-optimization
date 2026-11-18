# MPRALegNet for cell-selective enhancer design

[Notebook](../../examples/multiple-models.ipynb) · [Installation](../installation.md)

Combine the pretrained K562, HepG2 and WTC11 MPRALegNet predictors. Reference predictions put the three outputs on standardized scales before optimization.

## Load the predictors and set the objective

```python
import numpy as np
import torch
from bioseq_models import load_model
from sequence_optimization import CombinedScorer, MultiTaskObjective, generate_designs, random_seed_sequences

device = "cuda" if torch.cuda.is_available() else "cpu"
cells = ["K562", "HepG2", "WTC11"]
model = CombinedScorer(
    [load_model(f"agarwal_{cell.lower()}", device=device) for cell in cells],
    names=cells,
)
reference = model.score(random_seed_sequences(230, 32, seed=43)).scores
objective = MultiTaskObjective(
    high=["K562"], low=["HepG2", "WTC11"], aggregation="mean",
    centers=reference.mean(axis=0), scales=np.maximum(reference.std(axis=0), 1e-8),
)
seeds = random_seed_sequences(230, 4, seed=42)
```

## Optimize and inspect the candidates

```python
designs = generate_designs(
    model, length=230, initial=seeds, objective=objective,
    algorithms=["simulated_annealing"], num_candidates=4, seed=42, parallel=False,
)
designs[["activity_K562", "activity_HepG2", "activity_WTC11", "score"]]
```
