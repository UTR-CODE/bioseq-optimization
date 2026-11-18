# DeepSTARR for multi-task enhancer design

[Notebook](../../examples/multi-task.ipynb) · [Installation](../installation.md)

DeepSTARR predicts developmental and housekeeping enhancer activity. This example uses its two outputs to increase developmental activity and decrease housekeeping activity in 249-nt sequences.

## Load the model and set the objective

```python
import torch
from bioseq_models import load_model
from sequence_optimization import MultiTaskObjective, generate_designs, random_seed_sequences

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("deepstarr", device=device)
objective = MultiTaskObjective(high=["developmental"], low=["housekeeping"], aggregation="mean")
seeds = random_seed_sequences(249, 4, seed=42)
```

## Optimize and inspect the candidates

```python
designs = generate_designs(
    model, length=249, initial=seeds, objective=objective,
    algorithms=["simulated_annealing"], num_candidates=4, seed=42, parallel=False,
)
designs[["activity_developmental", "activity_housekeeping", "score"]]
```
