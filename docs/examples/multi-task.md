# Example 2: DeepSTARR multi-task enhancer design

[Home](../../README.md) · [Core API](../core-api.md) · [Notebook](../../examples/multi-task.ipynb)

Design 249-nt enhancers that increase DeepSTARR developmental activity and decrease housekeeping activity. One model returns both outputs in the order `developmental`, `housekeeping`.

## 1. Install the example dependencies

From the optimization repository:

```bash
python -m pip install -e '.[analysis,motifs]'
python -m pip install 'bioseq-models @ git+https://github.com/UTR-CODE/bioseq-models.git'
conda install -c conda-forge -c bioconda meme
```

The examples load pretrained predictors from [bioseq-models](https://github.com/UTR-CODE/bioseq-models). Checkpoints are downloaded on first use. The optimization functions accept sequence scorers or PyTorch modules; model loading belongs to this example. Use a new output directory for each run.

## 2. Load DeepSTARR and set opposing task directions

```python
import numpy as np
import pandas as pd
import torch
from bioseq_models import load_model
from sequence_optimization import (
    generate_designs, call_motifs, select_candidates,
    MultiTaskObjective, random_seed_sequences,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

scorer = load_model("deepstarr", device=DEVICE)
objective = MultiTaskObjective(
    high=["developmental"],
    low=["housekeeping"],
    aggregation="mean",
)
initial = random_seed_sequences(249, 100, seed=42)
initial_activity = scorer.score(initial)
print("Tasks:", scorer.task_names)
print(pd.DataFrame(initial_activity, columns=scorer.task_names).describe())
```

The score is `(developmental - housekeeping) / 2`. Increasing developmental prediction or decreasing housekeeping prediction improves this score. Swap the two task names to reverse the design direction.

## 3. Generate candidate sequences

```python
designs = generate_designs(
    scorer,
    length=249,
    initial=initial,
    objective=objective,
    num_candidates=100,
    seed=42,
    output_dir="outputs/deepstarr/designs",
)

timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
print(designs.groupby("algorithm").score.agg(["count", "median", "max"]))
```

All 11 optimizers start from the same 100 random sequences. Each runs to a plateau without a query budget and retains 100 candidates at the 90% observed-gain stage. The combined table contains 1,100 algorithm records; identical DNA sequences are combined during panel selection. Individual activities, objective scores, and algorithm provenance are retained.

## 4. Discover motifs for the design objective

```python
motifs = call_motifs(
    designs,
    model=scorer,
    min_seqlets=10,
    min_parents=3,
    output_dir="outputs/deepstarr/motifs",
)
print(motifs.support)
```

The function reuses the generation objective for ISM contributions, calls positive seqlets with tangermeme, discovers motifs with STREME against dinucleotide-shuffled controls, and retains motifs supported by at least 10 seqlets from at least 3 sequences. It saves the contributions, seqlets, support counts, and `catalog.meme`.

## 5. Select ten candidates

```python
unique = designs.drop_duplicates("sequence")
gate = float(unique.score.quantile(0.9))
selected = select_candidates(
    designs,
    motifs=motifs,
    top_n=10,
    min_score=gate,
    output_dir="outputs/deepstarr/top10",
)
print(selected[['selection_rank', 'sequence_id', 'activity_developmental', 'activity_housekeeping', 'score', 'present_motifs']])
```

The score gate retains the highest-scoring 10% of unique candidates. Within this pool, selection prioritizes motif composition diversity, then motif positions matched by the Hungarian algorithm, then prediction score for diversity ties. Motifs with fewer than 30 physical sites across the design pool are excluded by default. Outputs include `selected_sequences.csv`, FASTA, motif sites, distance matrices, and the selection trace.

For Hamming Max-Min selection, omit `motifs`:

```python
hamming_panel = select_candidates(
    designs, top_n=10, min_score=gate,
    output_dir="outputs/deepstarr/hamming_top10",
)
```

## 6. Inspect both activities and the specificity margin

```python
tasks = list(scorer.task_names)
activity_columns = [f"activity_{task}" for task in tasks]
comparison = pd.DataFrame({
    "starting_median": np.median(initial_activity, axis=0),
    "selected_median": selected[activity_columns].median().to_numpy(),
}, index=tasks)
comparison["median_change"] = comparison.selected_median - comparison.starting_median
print(comparison)

selected["developmental_minus_housekeeping"] = (
    selected.activity_developmental - selected.activity_housekeeping
)
print(selected[["sequence_id", *activity_columns, "developmental_minus_housekeeping"]])
```

The per-task changes show how the combined score was achieved. Contributions and discovered motifs describe the same developmental-high, housekeeping-low objective.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(initial_activity[:, 1], initial_activity[:, 0], alpha=0.4, label="Starting")
ax.scatter(designs.activity_housekeeping, designs.activity_developmental,
           s=10, alpha=0.25, label="Designed")
ax.scatter(selected.activity_housekeeping, selected.activity_developmental,
           s=60, marker="*", label="Selected")
ax.set(xlabel="Predicted housekeeping activity", ylabel="Predicted developmental activity")
ax.legend()
fig.tight_layout()
fig.savefig("outputs/deepstarr/top10/activity_tradeoff.png", dpi=200)
```

High-developmental, low-housekeeping sequences appear toward the upper left of this plot. For explicit activity thresholds, set `lower_bounds={"developmental": target_minimum}` and `upper_bounds={"housekeeping": off_target_maximum}` in `MultiTaskObjective`, using values in the model's activity units.
