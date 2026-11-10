# Enhancer design

[Home](../../README.md) · [Candidate generation](../candidate-generation.md) · [Notebook](../../examples/enhancer.ipynb)

Design 200-nt enhancers with high activity in a target cell type relative to two off-target cell types. This example names the model outputs K562, HepG2, and SKNSH and maximizes

\[
S(x)=a_{\mathrm{K562}}(x)-\max(a_{\mathrm{HepG2}}(x),a_{\mathrm{SKNSH}}(x)).
\]

Input: a trained PyTorch `model` returning these three activities in that order. The model accepts `[batch,4,200]`; context-dependent inputs can use the [input-transform interface](../pytorch-models.md).

## 1. Connect the activity model

Construct the two target-minus-off-target margins while retaining the three activity outputs.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_optimization import SequencePredictor, MultiTaskObjective, random_seed_sequences
from sequence_optimization.workflows import generate_candidates

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("outputs/enhancer")

def enhancer_outputs(activity):
    margins = activity[:, :1] - activity[:, 1:3]
    return torch.cat([activity, margins], dim=1)

scorer = SequencePredictor(
    model, device=DEVICE, batch_size=512,
    tasks=["K562", "HepG2", "SKNSH", "K562_minus_HepG2", "K562_minus_SKNSH"],
    output_transform=enhancer_outputs,
)
objective = MultiTaskObjective(high=[3, 4], aggregation="worst")
```

The minimum of the two margins equals the target-minus-maximum-off-target specificity. The transformation is used by both black-box and gradient algorithms.

## 2. Generate 1,100 candidates

```python
initial = random_seed_sequences(200, 100, seed=42)
candidates, timing = generate_candidates(
    scorer, 200, initial, objective,
    output_dir=OUT, num_candidates=100, diversity=5, seed=42,
)
print(timing[["algorithm", "seconds", "queries", "score_median"]])
```

Each algorithm runs to a plateau and contributes 100 candidates from its 90% observed-gain stage.

## 3. Summarize activity and specificity

```python
summary = candidates.groupby("algorithm").agg(
    target_activity=("activity_K562", "median"),
    specificity=("score", "median"),
)
print(summary.join(timing.set_index("algorithm")[["seconds", "queries"]]))
```

## 4. Discover motifs and scan the candidate pool

Follow [Motif discovery](../motif-discovery.md) with this scorer and candidate pool to create `outputs/motifs/catalog.meme`. Then scan the unique sequences:

```python
from sequence_optimization.motifs import read_meme, scan_fimo
catalog = read_meme("outputs/motifs/catalog.meme")
hits = scan_fimo(
    "outputs/motifs/catalog.meme",
    OUT / "unique_candidates.fa",
    "outputs/enhancer_fimo",
)
```

## 5. Select a motif-diverse Top10

Use a specificity threshold in the model's activity units. This example takes the upper quartile of candidate specificity as the quality threshold.

```python
from sequence_optimization.motifs import select_motif_panel
minimum_specificity = float(candidates.drop_duplicates("sequence").score.quantile(0.75))
selected, trace = select_motif_panel(
    candidates, hits, catalog,
    output_dir="outputs/enhancer_top10",
    top_n=10, min_score=minimum_specificity, min_sites=30,
    composition_tie=0.02, position_tie_bp=2.0,
)
print(selected[["selection_rank", "sequence_id", "score", "present_motifs", "algorithms"]])
```

Selection starts with the highest-specificity sequence, then maximizes motif composition diversity, followed by positional diversity, followed by specificity. See [Motif diversity](../motif-diversity.md) for the distance definitions.

## 6. Plot activities

```python
import matplotlib.pyplot as plt
values = selected[["activity_K562", "activity_HepG2", "activity_SKNSH"]].to_numpy()
fig, ax = plt.subplots(figsize=(5, 6))
im = ax.imshow(values, aspect="auto", cmap="viridis")
ax.set_xticks(range(3), ["K562", "HepG2", "SKNSH"])
ax.set_yticks(range(len(selected)), selected.sequence_id, fontsize=7)
fig.colorbar(im, ax=ax, label="Predicted activity")
fig.tight_layout()
fig.savefig("outputs/enhancer_top10/activity_heatmap.png", dpi=180)
```

## 7. Plot motif types and positions

```python
from matplotlib.patches import Rectangle
sites = pd.read_csv("outputs/enhancer_top10/motif_sites.csv")
motif_ids = sorted(sites.motif_id.unique())
colors = {m: plt.get_cmap("tab20")(i % 20) for i, m in enumerate(motif_ids)}
fig, ax = plt.subplots(figsize=(12, 5))
for y, row in enumerate(selected.itertuples()):
    ax.plot([0, 200], [y, y], color="lightgray", lw=2, zorder=0)
    for hit in sites[sites.sequence_name.eq(row.sequence_id)].itertuples():
        ax.add_patch(Rectangle((hit.start0, y-.25), hit.end0-hit.start0, .5,
                              facecolor=colors[hit.motif_id], alpha=.75))
        ax.text(hit.center0, y+.3, hit.motif_id, ha="center", fontsize=6, rotation=45)
ax.set_xlim(0, 200)
ax.set_ylim(-.5, len(selected))
ax.set_yticks(range(len(selected)), selected.sequence_id, fontsize=7)
ax.set_xlabel("Enhancer position (bp)")
fig.tight_layout()
fig.savefig("outputs/enhancer_top10/motif_positions.png", dpi=180)
```

Each rectangle shows one FIMO footprint and its catalog motif ID. The exported CSV and FASTA contain the selected sequences, predictions, and algorithm provenance.
