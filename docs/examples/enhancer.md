# Malinois for K562 enhancer design

[Installation](../installation.md)

Design 200-nt enhancers that favor K562 activity over HepG2 and SK-N-SH.

## 1. Load Malinois

```python
from pathlib import Path

import torch
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, random_seed_sequences, select_candidates

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = load_model("malinois", device=device, batch_size=1024)
run = Path("outputs/malinois_k562")
```

Malinois adds fixed flanks when it encodes DNA strings. Use the same flanks for gradient-based scoring.

```python
context = model.encoder(["A" * 200]).transpose(1, 2)

def add_flanks(probabilities):
    batch = len(probabilities)
    left = context[:, :200].to(probabilities).expand(batch, -1, -1)
    right = context[:, -200:].to(probabilities).expand(batch, -1, -1)
    return torch.cat((left, probabilities, right), dim=1).transpose(1, 2)

model.soft_encoder = add_flanks

def specificity(y):
    return y["K562_log2FC"] - torch.maximum(y["HepG2_log2FC"], y["SKNSH_log2FC"])
```

## 2. Generate candidates

```python
seeds = random_seed_sequences(200, 100, seed=42)
designs = generate_designs(
    model, length=200, initial=seeds, objective=specificity, seed=42,
    output_dir=run / "designs",
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
