# DREAM LegNet for yeast promoter design

[Notebook](../../examples/dream-legnet-promoter.ipynb) · [Installation](../installation.md)

Optimize an 80-nt yeast promoter insert for predicted activity. Keep the assay context fixed and exclude runs of six or more identical bases.

## 1. Load the model

```python
import re
from pathlib import Path

import torch
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, random_seed_sequences, select_candidates

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dream = load_model("dream_legnet", variable_region_only=True, device=device, batch_size=2048)
suffix = "GGTTACGGCTGTT"
context = dream.encoder(["A" * 80 + suffix])

class PromoterScorer:
    task_names = dream.task_names
    device = dream.device

    def score(self, sequences):
        return dream.score([sequence + suffix for sequence in sequences])

    def score_soft(self, probabilities):
        batch = len(probabilities)
        fixed = context.to(probabilities).expand(batch, -1, -1)
        insert = probabilities[:, :, [0, 2, 1, 3]].transpose(1, 2)
        dna = torch.cat((fixed[:, :4, :57], insert, fixed[:, :4, -13:]), dim=2)
        inputs = torch.cat((dna, fixed[:, 4:, :]), dim=1)
        return dream.model(inputs).reshape(-1, 1)

model = PromoterScorer()
run = Path("outputs/dream_legnet_promoter")
```

## 2. Generate candidates

```python
def no_homopolymer6(sequences):
    return [re.search(r"([ACGT])\1{5}", sequence) is None for sequence in sequences]

seeds = random_seed_sequences(80, 100, seed=42)
designs = generate_designs(
    model, length=80, initial=seeds, objective="maximize",
    constraint=no_homopolymer6, output_dir=run / "designs",
)
designs.groupby("algorithm")["score"].agg(["size", "median", "max"])
```

## 3. Discover motifs

```python
motifs = call_motifs(designs, model, output_dir=run / "motifs")
motifs.support.query("retained")[["motif_id", "seqlets", "parents", "sites_on_unique_DNA"]]
```

## 4. Select a panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.2))
selected = select_candidates(designs, motifs=motifs, top_n=10, min_score=gate, output_dir=run / "selected")
selected[["selection_rank", "sequence", "score", "present_motifs"]]
```
