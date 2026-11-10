# Integrating multiple single-output models

[Home](../../README.md) · [Core API](../core-api.md) · [Notebook](../../examples/multiple-models.ipynb)

Use independent PyTorch predictors `model_a`, `model_b`, and `model_c`, each returning one activity for the same design region.

## 1. Combine the predictors

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_optimization import generate_designs, call_motifs, select_candidates
from sequence_optimization import SequencePredictor, CombinedScorer, MultiTaskObjective
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
scorer = CombinedScorer(
    [SequencePredictor(model_a, device=DEVICE),
     SequencePredictor(model_b, device=DEVICE),
     SequencePredictor(model_c, device=DEVICE)],
    names=["assay_a", "assay_b", "assay_c"],
)
objective = MultiTaskObjective(high="all", aggregation="mean")
```

This objective maximizes the mean prediction of the three models. Each child scorer can use its own input transform or fixed context.

## 2. Generate candidate designs

```python
designs = generate_designs(
    scorer, length=200, objective=objective,
    num_candidates=100, seed=42, output_dir="outputs/multiple_models",
)
timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
```

The 11 optimizers share a starting population, run to a plateau, and each contribute 100 designs at the 90% observed-gain stage. The returned table contains sequences, individual activity predictions, scores, and algorithm provenance.

## 3. Call motifs from the designs

```python
motifs = call_motifs(
    designs, model=scorer, output_dir="outputs/multiple_models_motifs",
)
print(motifs.support)
```

This calculates contributions for the optimization objective, extracts positive seqlets, and discovers supported PWMs. The catalog is saved as `catalog.meme`.

## 4. Select a motif-diverse panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.9))
selected = select_candidates(
    designs, top_n=10, motifs=motifs, min_score=gate,
    output_dir="outputs/multiple_models_top10",
)
print(selected[["selection_rank", "activity_assay_a", "activity_assay_b", "activity_assay_c", "score", "present_motifs"]])
```

The selector scans the catalog onto the designs and prioritizes motif composition, matched motif positions, and prediction score. CSV, FASTA, motif sites, distance matrices, and the selection trace are exported together.

## 5. Hamming selection without motifs

For a sequence-distance panel, omit the `motifs` argument. This mode also works directly after design generation, independently of motif discovery.

```python
hamming_panel = select_candidates(
    designs, top_n=10, min_score=gate,
    output_dir="outputs/multiple_models_hamming",
)
print(hamming_panel.attrs["selection_method"])  # hamming
```

## 6. Alternative integration objectives

Weighted ensemble:

```python
weighted_objective = MultiTaskObjective(
    high="all", aggregation="mean", weights=[0.5, 0.3, 0.2],
)
```

For different prediction scales, estimate fixed reference statistics and optimize standardized outputs:

```python
from sequence_optimization import random_seed_sequences
reference = random_seed_sequences(200, 1000, seed=43)
reference_activity = scorer.score(reference).scores
normalized_objective = MultiTaskObjective(
    high="all", aggregation="mean",
    centers=reference_activity.mean(axis=0).tolist(),
    scales=np.maximum(reference_activity.std(axis=0), 1e-8).tolist(),
)
```

For one model high and the others low:

```python
selective_objective = MultiTaskObjective(
    high=["assay_a"], low=["assay_b", "assay_c"], aggregation="mean",
)
```

Pass the chosen objective to `generate_designs`. `call_motifs` automatically reuses that objective when computing contributions.
