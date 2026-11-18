# MotifSift API

## Choose an objective

For a single output, pass `objective="maximize"` or `objective="minimize"`.

For multiple outputs, name the outputs to increase and decrease. Use `weights` when some outputs matter more. Omitted weights default to 1. Model outputs outside `high` and `low` are excluded from the score.

```python
from MotifSift import MultiTaskObjective

objective = MultiTaskObjective(
    high=["housekeeping"],
    low=["developmental"],
    weights={"housekeeping": 2, "developmental": 1},
)
```

This maximizes `(2 × housekeeping − developmental) / 3` in the model's prediction units. Omit `weights` to give both outputs equal weight.

For a custom score, pass a function that receives a dictionary of named PyTorch prediction tensors and returns one score per sequence. The optimizer maximizes the returned score. Use tensor operations so gradient-based optimizers can differentiate through it.

```python
import torch


def k562_specificity(y):
    return y["K562_log2FC"] - torch.maximum(y["HepG2_log2FC"], y["SKNSH_log2FC"])


def housekeeping_specificity(y):
    return y["housekeeping"] - y["developmental"]
```

Use `generate_designs(model, length=200, objective=k562_specificity)` with the [Malinois enhancer model](examples/enhancer.md). `designs` keeps the individual model predictions alongside the composite `score`.

If the outputs come from separate predictors, combine them with `CombinedScorer({"K562": k562_model, "HepG2": hepg2_model})`. The dictionary keys become the output names used by the objective.

## Generate, discover, and select

```python
from MotifSift import call_motifs, generate_designs, select_candidates

designs = generate_designs(model, length=200, objective=objective)
motifs = call_motifs(designs, model)
panel = select_candidates(designs, motifs=motifs, top_n=10)
```

- `generate_designs` returns a pandas DataFrame. By default, eleven optimizers each retain up to 100 candidates at 90% of the observed gain. Use `num_candidates` to change the count, `gain_fractions=(0.5, 0.9)` to return both stages, or `score_threshold` to set a target score.
- `call_motifs` uses the design objective for attribution. Use `reverse_complement=False` for RNA. When reloading designs from a CSV, pass the original `objective` explicitly to restore it.
- `select_candidates` returns a DataFrame of the selected sequences. Use `top_n` for panel size and `min_score` to enforce a minimum score.

Each step accepts an optional `output_dir` for saving its results. See [PyTorch models](pytorch-models.md), [motif discovery](motif-discovery.md), and [panel selection](motif-diversity.md).
