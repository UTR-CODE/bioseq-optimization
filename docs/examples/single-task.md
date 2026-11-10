# Example 1: Optimus 5-Prime single-task design

[Home](../../README.md) · [Core API](../core-api.md) · [Notebook](../../examples/single-task.ipynb)

Design 50-nt 5′UTRs with high predicted mean ribosome load (MRL). Optimus 5-Prime returns one output, `mrl`, so the optimization score is the predicted MRL itself.

## 1. Install the example dependencies

From the optimization repository:

```bash
python -m pip install -e '.[analysis,motifs]'
python -m pip install 'bioseq-models @ git+https://github.com/UTR-CODE/bioseq-models.git'
conda install -c conda-forge -c bioconda meme
```

The examples load pretrained predictors from [bioseq-models](https://github.com/UTR-CODE/bioseq-models). Checkpoints are downloaded on first use. The optimization functions accept sequence scorers or PyTorch modules; model loading belongs to this example. Use a new output directory for each run.

## 2. Load the predictor and define the objective

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

scorer = load_model("optimus5prime", device=DEVICE)
objective = ["maximize"]
initial = random_seed_sequences(50, 100, seed=42)
initial_activity = scorer.score(initial)
print("Tasks:", scorer.task_names)  # ('mrl',)
print("Starting median MRL:", np.median(initial_activity[:, 0]))
```

The adapter supplies discrete sequence scoring and differentiable scoring for the gradient optimizers. It preserves the checkpoint's MRL output scaling. Set `objective = ["minimize"]` to design low-MRL UTRs.

## 3. Generate candidate sequences

```python
designs = generate_designs(
    scorer,
    length=50,
    initial=initial,
    objective=objective,
    num_candidates=100,
    seed=42,
    output_dir="outputs/optimus5prime/designs",
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
    output_dir="outputs/optimus5prime/motifs",
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
    output_dir="outputs/optimus5prime/top10",
)
print(selected[['selection_rank', 'sequence_id', 'activity_mrl', 'score', 'present_motifs']])
```

The score gate retains the highest-scoring 10% of unique candidates. Within this pool, selection prioritizes motif composition diversity, then motif positions matched by the Hungarian algorithm, then prediction score for diversity ties. Motifs with fewer than 30 physical sites across the design pool are excluded by default. Outputs include `selected_sequences.csv`, FASTA, motif sites, distance matrices, and the selection trace.

For Hamming Max-Min selection, omit `motifs`:

```python
hamming_panel = select_candidates(
    designs, top_n=10, min_score=gate,
    output_dir="outputs/optimus5prime/hamming_top10",
)
```

## 6. Compare the selected panel with the starting sequences

```python
comparison = pd.DataFrame({
    "starting_median": np.median(initial_activity, axis=0),
    "selected_median": selected[["activity_mrl"]].median().to_numpy(),
}, index=["mrl"])
comparison["median_change"] = comparison.selected_median - comparison.starting_median
print(comparison)

selected["upstream_ATG_count"] = selected.sequence.map(
    lambda s: sum(s[i:i+3] == "ATG" for i in range(len(s)-2))
)
print(selected[["sequence_id", "activity_mrl", "upstream_ATG_count", "sequence"]])
```

The reported comparison is in predicted MRL units. The UTR annotation records upstream start codons alongside the final sequences. To inspect motif coordinates, read `selected.attrs["motif_instances"]`.
