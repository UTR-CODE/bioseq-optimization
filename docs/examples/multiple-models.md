# Integrating multiple single-output models

[Home](../../README.md) · [Notebook](../../examples/multiple-models.ipynb) · [Single task](single-task.md) · [Multi-task](multi-task.md)

Evaluate each sequence with several independent PyTorch models, then optimize their combined predictions. Each model returns one activity. The models may represent replicate predictors of the same activity or different sequence properties.

## 1. Wrap each model

`model_a`, `model_b`, and `model_c` are trained PyTorch modules. Each receives the same 200-nt design region and returns `[batch]` or `[batch,1]`.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_optimization import SequencePredictor, CombinedScorer, MultiTaskObjective, random_seed_sequences

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LENGTH = 200
scorer_a = SequencePredictor(model_a, tasks=["assay_a"], device=DEVICE, batch_size=512)
scorer_b = SequencePredictor(model_b, tasks=["assay_b"], device=DEVICE, batch_size=512)
scorer_c = SequencePredictor(model_c, tasks=["assay_c"], device=DEVICE, batch_size=512)

combined = CombinedScorer(
    [scorer_a, scorer_b, scorer_c], names=["assay_a", "assay_b", "assay_c"],
)
initial = random_seed_sequences(LENGTH, 100, seed=42)
print(pd.DataFrame(combined.score(initial[:4]).scores, columns=combined.task_names))
```

`CombinedScorer` evaluates all models on the same sequences and concatenates their activity columns. Each child scorer can use its own input transform, fixed context, or device.

## 2. Choose the integration objective

For models predicting activities on a common scale, maximize their mean:

```python
objective = MultiTaskObjective(high="all", aggregation="mean")
```

This defines `S(x) = (a(x) + b(x) + c(x)) / 3`. The raw prediction from every model remains in the candidate table.

## 3. Generate the combined candidate pool

```python
from sequence_optimization.workflows import generate_candidates
candidates, timing = generate_candidates(
    combined, LENGTH, initial, objective,
    output_dir="outputs/multiple_models", num_candidates=100,
    diversity=5, seed=42,
)
print(candidates[["algorithm", "activity_assay_a", "activity_assay_b", "activity_assay_c", "score"]].head())
print(timing[["algorithm", "seconds", "queries"]])
```

The 11 algorithms optimize the same combined objective and contribute 100 candidates each. Gradients from all child predictors propagate to the shared sequence probabilities.

## 4. Select and export the panel

```python
from sequence_optimization.workflows import deduplicate_candidates, select_sequence_panel, write_fasta
unique, provenance = deduplicate_candidates(candidates)
gate = float(unique.score.quantile(0.9))
selected, distances = select_sequence_panel(candidates, top_n=10, min_score=gate)

panel = Path("outputs/multiple_models_top10")
panel.mkdir(parents=True, exist_ok=True)
selected.to_csv(panel / "selected_sequences.csv", index=False)
write_fasta(panel / "selected_sequences.fa", selected.sequence_id, selected.sequence)
print(selected[["selection_rank", "activity_assay_a", "activity_assay_b", "activity_assay_c", "score", "algorithms"]])
```

## 5. Alternative integration objectives

### Weighted ensemble

For predictors of the same activity, assign model weights:

```python
weighted_objective = MultiTaskObjective(
    high="all", aggregation="mean", weights=[0.5, 0.3, 0.2],
)
```

### Different properties on different scales

Estimate a fixed reference mean and scale for each predictor, then maximize their standardized predictions:

```python
reference = random_seed_sequences(LENGTH, 1000, seed=43)
reference_activity = combined.score(reference).scores
normalized_objective = MultiTaskObjective(
    high="all", aggregation="mean",
    centers=reference_activity.mean(axis=0).tolist(),
    scales=np.maximum(reference_activity.std(axis=0), 1e-8).tolist(),
)
```

The reference statistics remain fixed throughout optimization.

### One model high, the other models low

Treat the combined model columns as named tasks:

```python
selective_objective = MultiTaskObjective(
    high=["assay_a"], low=["assay_b", "assay_c"], aggregation="mean",
)
```

Pass the selected objective to `generate_candidates` with a new output directory. All three configurations use the same candidate-generation and [motif-diversity selection](../motif-diversity.md) interfaces.
