# Enhancer design

[Home](../../README.md) · [Core API](../core-api.md) · [Notebook](../../examples/enhancer.ipynb)

Design 200-nt enhancers maximizing `K562 - max(HepG2, SKNSH)` with a PyTorch `model` returning the three activities.

## 1. Define enhancer specificity

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_optimization import generate_designs, call_motifs, select_candidates
from sequence_optimization import SequencePredictor, MultiTaskObjective
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def enhancer_outputs(y):
    return torch.cat([y, y[:, :1] - y[:, 1:]], dim=1)

scorer = SequencePredictor(
    model, device=DEVICE,
    tasks=["K562", "HepG2", "SKNSH", "K562_minus_HepG2", "K562_minus_SKNSH"],
    output_transform=enhancer_outputs,
)
objective = MultiTaskObjective(high=[3, 4], aggregation="worst")
```

The minimum pairwise target-minus-off-target margin equals the desired specificity.

## 2. Generate candidate designs

```python
designs = generate_designs(
    scorer, length=200, objective=objective,
    num_candidates=100, seed=42, output_dir="outputs/enhancer",
)
timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
```

The 11 optimizers share a starting population, run to a plateau, and each contribute 100 designs at the 90% observed-gain stage. The returned table contains sequences, individual activity predictions, scores, and algorithm provenance.

## 3. Call motifs from the designs

```python
motifs = call_motifs(
    designs, model=scorer, output_dir="outputs/enhancer_motifs",
)
print(motifs.support)
```

This calculates contributions for the optimization objective, extracts positive seqlets, and discovers supported PWMs. The catalog is saved as `catalog.meme`.

## 4. Select a motif-diverse panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.9))
selected = select_candidates(
    designs, top_n=10, motifs=motifs, min_score=gate,
    output_dir="outputs/enhancer_top10",
)
print(selected[["selection_rank", "activity_K562", "activity_HepG2", "activity_SKNSH", "score", "present_motifs"]])
```

The selector scans the catalog onto the designs and prioritizes motif composition, matched motif positions, and prediction score. CSV, FASTA, motif sites, distance matrices, and the selection trace are exported together.

## 5. Hamming selection without motifs

For a sequence-distance panel, omit the `motifs` argument. This mode also works directly after design generation, independently of motif discovery.

```python
hamming_panel = select_candidates(
    designs, top_n=10, min_score=gate,
    output_dir="outputs/enhancer_hamming",
)
print(hamming_panel.attrs["selection_method"])  # hamming
```

## 6. Plot the selected activities and motif positions

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(5, 6))
values = selected[["activity_K562", "activity_HepG2", "activity_SKNSH"]].to_numpy()
im = ax.imshow(values, aspect="auto", cmap="viridis")
ax.set_xticks(range(3), ["K562", "HepG2", "SKNSH"])
ax.set_yticks(range(len(selected)), selected.sequence_id, fontsize=7)
fig.colorbar(im, ax=ax, label="Predicted activity")
fig.tight_layout()
fig.savefig("outputs/enhancer_top10/activity_heatmap.png", dpi=180)

sites = pd.read_csv("outputs/enhancer_top10/motif_sites.csv")
colors = {m: plt.get_cmap("tab20")(i % 20) for i, m in enumerate(sorted(sites.motif_id.unique()))}
fig, ax = plt.subplots(figsize=(12, 5))
for y, row in enumerate(selected.itertuples()):
    ax.plot([0, 200], [y, y], color="lightgray", lw=2)
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
