# Optimus 5-Prime for 5′ UTR design

[Notebook](../../examples/single-task.ipynb) · [Installation](../installation.md)

This example maximizes predicted mean ribosome load for a 50-nt 5′ UTR. The first model load downloads the pretrained checkpoint.

## Load the predictor

```python
import torch
from bioseq_models import load_model
from sequence_optimization import generate_designs, random_seed_sequences

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("optimus5prime", device=device)
seeds = random_seed_sequences(50, 4, seed=42)
```

## Optimize and inspect the candidates

```python
designs = generate_designs(
    model, length=50, initial=seeds, objective=["maximize"],
    algorithms=["simulated_annealing"], num_candidates=4, seed=42, parallel=False,
)
designs[["sequence", "activity_mrl"]]
```
