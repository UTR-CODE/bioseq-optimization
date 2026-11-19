# Optimize enhancer specificity with Malinois

[Notebook](../../examples/enhancer.ipynb) · [Installation](../installation.md)

Design 200-nt enhancers for K562. The Malinois predictor adds two fixed 200-nt assay flanks, and its network scores the resulting 600-nt construct. The objective subtracts the higher off-target prediction from the K562 prediction.

## 1. Load the model and define the objective

```python
import os
from pathlib import Path

import torch
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, random_seed_sequences, select_candidates

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
predictor = load_model("malinois", device=device)
# Extract the fixed flanks from a placeholder construct.
reference = predictor.encoder(["A" * 200]).transpose(1, 2)
left, right = reference[:, :200], reference[:, -200:]


class EnhancerScorer:
    task_names = predictor.task_names
    device = predictor.device

    def score(self, sequences):
        return predictor.score(sequences)

    def score_soft(self, probabilities):
        n = len(probabilities)
        construct = torch.cat([
            left.to(probabilities).expand(n, -1, -1),
            probabilities,
            right.to(probabilities).expand(n, -1, -1),
        ], dim=1)
        return predictor.model(construct.transpose(1, 2))


model = EnhancerScorer()


def specificity(y):
    return y["K562_log2FC"] - torch.maximum(y["HepG2_log2FC"], y["SKNSH_log2FC"])


run = Path("outputs/malinois_k562")
os.makedirs(run, exist_ok=True)
```

The scoring adapter adds the same fixed flanks to soft inputs as the pretrained predictor uses for sequence strings.

## 2. Generate candidates

The saved tables are from a full rerun with base seed 42 and FP32 inference. That run used the assay’s archived GPU batch sizes (1,024 discrete and 128 differentiable).

```python
seeds = random_seed_sequences(200, 100, seed=42)
designs = generate_designs(
    model, length=200, initial=seeds, objective=specificity,
    output_dir=run / "designs",
)
designs.groupby("algorithm")["score"].agg(["size", "median", "max"])
```

## 3. Discover motifs

Discover motifs from nucleotide contributions to the specificity objective.

```python
motifs = call_motifs(designs, model, output_dir=run / "motifs")
motifs.support.query("retained")[["motif_id", "seqlets", "parents", "sites_on_unique_DNA"]]
```

## 4. Select a panel

Use the 20th percentile of the unique candidates’ scores as the minimum score for the panel.

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.2))
panel = select_candidates(designs, motifs=motifs, top_n=10, min_score=gate, output_dir=run / "selected")
panel[["selection_rank", "sequence", "score", "present_motifs"]]
```
