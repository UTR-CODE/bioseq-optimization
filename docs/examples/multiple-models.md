# Example 3: Multiple single-task models for cell-selective design

[Home](../../README.md) · [Core API](../core-api.md) · [Notebook](../../examples/multiple-models.ipynb)

Design 230-nt enhancers with high K562 activity and low HepG2 and WTC11 activity. Three independently trained Agarwal lentiMPRA predictors each return one activity. `CombinedScorer` evaluates the same sequence with all three and assembles a three-column prediction matrix.

## 1. Install the example dependencies

From the optimization repository:

```bash
python -m pip install -e '.[analysis,motifs]'
python -m pip install 'bioseq-models @ git+https://github.com/UTR-CODE/bioseq-models.git'
conda install -c conda-forge -c bioconda meme
```

The examples load pretrained predictors from [bioseq-models](https://github.com/UTR-CODE/bioseq-models). Checkpoints are downloaded on first use. The optimization functions accept sequence scorers or PyTorch modules; model loading belongs to this example. Use a new output directory for each run.

## 2. Load the three predictors and define the joint objective

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
from sequence_optimization import CombinedScorer

scorer = CombinedScorer(
    [load_model("agarwal_k562", device=DEVICE),
     load_model("agarwal_hepg2", device=DEVICE),
     load_model("agarwal_wtc11", device=DEVICE)],
    names=["K562", "HepG2", "WTC11"],
)

# Estimate fixed scales on the same reference DNA for all predictors.
reference = random_seed_sequences(230, 1000, seed=43)
reference_activity = scorer.score(reference).scores
centers = reference_activity.mean(axis=0)
scales = np.maximum(reference_activity.std(axis=0), 1e-8)
objective = MultiTaskObjective(
    high=["K562"],
    low=["HepG2", "WTC11"],
    centers=centers.tolist(),
    scales=scales.tolist(),
    aggregation="mean",
)

initial = random_seed_sequences(230, 100, seed=42)
initial_activity = scorer.score(initial).scores
print(pd.DataFrame({"center": centers, "scale": scales}, index=scorer.task_names))
```

For task `t`, let `z_t = (activity_t - center_t) / scale_t`. The score is `(z_K562 - z_HepG2 - z_WTC11) / 3`. Reference statistics stay fixed throughout optimization and motif calling. This gives each task one reference-standard-deviation unit of influence. Individual activity columns remain in each model's original output units.

Each cell-specific predictor is a single-output cross-validation ensemble. The three predictors represent distinct cell types. Their adapters apply their own sequence encoding and support gradients through the combined score.

## 3. Generate candidate sequences

```python
designs = generate_designs(
    scorer,
    length=230,
    initial=initial,
    objective=objective,
    num_candidates=100,
    seed=42,
    output_dir="outputs/agarwal_selective/designs",
)

timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
print(designs.groupby("algorithm").score.agg(["count", "median", "max"]))
```

All 11 optimizers start from the same 100 random sequences and run in independent CPU processes. A shared scoring service batches their prediction and gradient requests for the model. Each runs to a plateau without a query budget and retains 100 candidates at the 90% observed-gain stage. The combined table contains 1,100 algorithm records; identical DNA sequences are combined during panel selection. Individual activities, objective scores, and algorithm provenance are retained.

## 4. Discover motifs for the design objective

```python
motifs = call_motifs(
    designs,
    model=scorer,
    min_seqlets=10,
    min_parents=3,
    output_dir="outputs/agarwal_selective/motifs",
)
print(motifs.support)
```

The function reuses the generation objective for ISM contributions, calls positive seqlets with tangermeme, discovers motifs with official STREME against dinucleotide-shuffled controls, and retains motifs with FIMO support from at least 10 seqlets and at least 3 parent sequences. It saves the contributions, seqlets, support counts, and `catalog.meme`.

The exported MEME file can also be passed as `motifs="outputs/agarwal_selective/motifs/catalog.meme"` in the selection call below. To use your own motif-discovery method, skip this step and set `motifs="my_motifs.meme"` instead.

## 5. Select ten candidates

```python
unique = designs.drop_duplicates("sequence")
gate = float(unique.score.quantile(0.8))
selected = select_candidates(
    designs,
    motifs=motifs,
    top_n=10,
    min_score=gate,
    output_dir="outputs/agarwal_selective/top10",
)
print(selected[['selection_rank', 'sequence_id', 'activity_K562', 'activity_HepG2', 'activity_WTC11', 'score', 'present_motifs']])
```

The score gate retains the highest-scoring 20% of unique candidates. Within this pool, selection prioritizes motif composition diversity, then motif positions matched by the Hungarian algorithm, then prediction score for diversity ties. Motifs with fewer than 30 physical sites across the design pool are excluded by default. Outputs include `selected_sequences.csv`, FASTA, motif sites, distance matrices, and the selection trace.

For Hamming Max-Min selection, omit `motifs`:

```python
hamming_panel = select_candidates(
    designs, top_n=10, min_score=gate,
    output_dir="outputs/agarwal_selective/hamming_top10",
)
```

## 6. Report target and off-target changes separately

```python
tasks = list(scorer.task_names)
activity_columns = [f"activity_{task}" for task in tasks]
comparison = pd.DataFrame({
    "starting_median": np.median(initial_activity, axis=0),
    "selected_median": selected[activity_columns].median().to_numpy(),
}, index=tasks)
comparison["median_change"] = comparison.selected_median - comparison.starting_median
comparison["change_in_reference_sd"] = comparison.median_change / scales
print(comparison)

standardized = (selected[activity_columns].to_numpy() - centers) / scales
selected["K562_vs_strongest_offtarget_z"] = (
    standardized[:, 0] - standardized[:, 1:].max(axis=1)
)
print(selected[["sequence_id", *activity_columns, "K562_vs_strongest_offtarget_z"]])
```

The table reports all three predicted activities and a standardized margin against the strongest off-target. It complements the mean objective used for optimization.

To prioritize the weakest target or off-target margin, use the smallest signed standardized activity as the objective:

```python
worst_objective = MultiTaskObjective(
    high=["K562"], low=["HepG2", "WTC11"],
    centers=centers.tolist(), scales=scales.tolist(),
    aggregation="worst",
)
```

Pass this as `objective` to `generate_designs` for a separate run. Its score is `min(z_K562, -z_HepG2, -z_WTC11)`. `call_motifs` then uses that same objective automatically. For another target cell, change `high` and `low` while retaining the scorer's task names.

To use independently trained PyTorch models from another study, replace the loaded adapters with `SequencePredictor(model, tasks=["activity"], device=DEVICE)` for each single-output model. Input transforms can supply model-specific layouts or contexts; see [PyTorch model interface](../pytorch-models.md).

For standalone Python scripts, place model loading and workflow calls under `if __name__ == "__main__":`. Notebook cells call the functions directly. See [Parallel scoring](../parallel-scoring.md) for batch-size controls and utilization metrics.
