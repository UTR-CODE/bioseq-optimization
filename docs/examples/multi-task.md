# Multi-task design: one output high, the others low

[Home](../../README.md) · [Notebook](../../examples/multi-task.ipynb) · [Single task](single-task.md) · [Multiple models](multiple-models.md)

Use a model predicting multiple activities for each sequence. This example has three outputs: `target`, `off1`, and `off2`. The goal is to increase target activity and decrease both off-target activities.

## 1. Connect the multi-output model

The trained PyTorch `model` returns `[batch,3]` in the named task order.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_optimization import SequencePredictor, MultiTaskObjective, random_seed_sequences

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LENGTH = 200
TASKS = ["target", "off1", "off2"]
scorer = SequencePredictor(model, tasks=TASKS, device=DEVICE, batch_size=512)
initial = random_seed_sequences(LENGTH, 100, seed=42)
```

## 2. Specify which output should be high or low

```python
objective = MultiTaskObjective(
    high=["target"], low=["off1", "off2"], aggregation="mean",
)
activities = scorer.score(initial[:4]).scores
fitness = objective.scalarize(activities, scorer.task_names)
print(pd.DataFrame(activities, columns=TASKS).assign(score=fitness))
```

With equal task scales, the score is

\[
S(x)=\frac{a_{\mathrm{target}}(x)-a_{\mathrm{off1}}(x)-a_{\mathrm{off2}}(x)}{3}.
\]

The equivalent directional form is `MultiTaskObjective(directions=["maximize", "minimize", "minimize"])`. For a model with more outputs, list the additional off-target names in `low`.

## 3. Generate 1,100 candidates

```python
from sequence_optimization.workflows import generate_candidates
candidates, timing = generate_candidates(
    scorer, LENGTH, initial, objective,
    output_dir="outputs/multi_task", num_candidates=100,
    diversity=5, seed=42,
)
print(timing[["algorithm", "seconds", "queries", "score_median"]])
print(candidates[["algorithm", "activity_target", "activity_off1", "activity_off2", "score"]].head())
```

Every candidate retains all three raw predictions and the scalar optimization score.

## 4. Select and export candidates

```python
from sequence_optimization.workflows import deduplicate_candidates, select_sequence_panel, write_fasta
unique, provenance = deduplicate_candidates(candidates)
gate = float(unique.score.quantile(0.9))
selected, distances = select_sequence_panel(candidates, top_n=10, min_score=gate)
selected["specificity"] = selected.activity_target - selected[["activity_off1", "activity_off2"]].max(axis=1)

panel = Path("outputs/multi_task_top10")
panel.mkdir(parents=True, exist_ok=True)
selected.to_csv(panel / "selected_sequences.csv", index=False)
write_fasta(panel / "selected_sequences.fa", selected.sequence_id, selected.sequence)
print(selected[["selection_rank", "activity_target", "activity_off1", "activity_off2", "score", "specificity"]])
```

## 5. Target thresholds

To optimize the weakest margin relative to desired activity thresholds, use a bounded worst-task objective. For outputs on a probability scale, an example is target activity at least 0.8 and each off-target activity at most 0.2:

```python
bounded_objective = MultiTaskObjective(
    high=["target"], low=["off1", "off2"],
    lower_bounds={"target": 0.8},
    upper_bounds={"off1": 0.2, "off2": 0.2},
    aggregation="worst",
)
```

Its score is `min(target - 0.8, 0.2 - off1, 0.2 - off2)`. A nonnegative score meets all three thresholds. Pass `bounded_objective` to the candidate-generation call to use this design criterion.

## 6. Direct target specificity

For the objective `target - max(off1, off2)`, append pairwise target-minus-off-target contrasts and maximize their minimum:

```python
def activity_and_contrasts(y):
    return torch.cat([y, y[:, :1] - y[:, 1:]], dim=1)

specificity_scorer = SequencePredictor(
    model, device=DEVICE, batch_size=512,
    tasks=["target", "off1", "off2", "target_minus_off1", "target_minus_off2"],
    output_transform=activity_and_contrasts,
)
specificity_objective = MultiTaskObjective(
    high=["target_minus_off1", "target_minus_off2"], aggregation="worst",
)
```

Use this scorer and objective in the same optimization call, with a new output directory. The [enhancer workflow](enhancer.md) applies this criterion and then selects a motif-diverse panel.
