# DeepSTARR for multi-task enhancer design

[Notebook](../../examples/multi-task.ipynb) · [Installation](../installation.md)

Decrease predicted developmental activity and increase predicted housekeeping activity in 249-nt enhancers. Install MEME Suite with the notebook dependencies. The first model load downloads the pretrained checkpoint.

## 1. Load the model

```python
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
import torch
from bioseq_models import load_model
from MotifSift import MultiTaskObjective, call_motifs, generate_designs, select_candidates

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("deepstarr", device=device)
objective = MultiTaskObjective(high=["housekeeping"], low=["developmental"])
seeds = pd.read_csv("data/enhancer_initial_sequences.csv")["deepstarr"].tolist()
run = Path(mkdtemp(prefix="motifsift-example-"))
```

## 2. Generate candidates

Run the eleven optimizers and keep 100 candidates per optimizer. The saved tables show the previous full run. New runs use the current MotifSift defaults.

```python
designs = generate_designs(
    model, length=249, initial=seeds, objective=objective,
    output_dir=run / "designs",
)
designs.groupby("algorithm")["score"].agg(["size", "median", "max"])
```

## 3. Discover motifs

Extract positive seqlets from in silico mutagenesis scores and discover motifs with XSTREME.

```python
motifs = call_motifs(
    designs, model=model,
    output_dir=run / "motifs",
)
motifs.support.query("retained")[["motif_id", "seqlets", "parents", "sites_on_unique_DNA"]]
```

## 4. Select a panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.2))
selected = select_candidates(designs, motifs=motifs, top_n=10, min_score=gate, output_dir=run / "selected")
selected[["selection_rank", "sequence", "score", "present_motifs"]]
```
