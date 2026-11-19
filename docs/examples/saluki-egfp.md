# Saluki for synonymous eGFP CDS design

[Notebook](../../examples/saluki-egfp.ipynb) · [Installation](../installation.md)

Score a single-exon eGFP transcript with fixed UTRs. Optimize the CDS with synonymous substitutions and retain the terminal stop codon. Install MEME Suite with the notebook dependencies. The first model load downloads the pretrained checkpoint.

## 1. Load the model

```python
from pathlib import Path
from tempfile import mkdtemp

import torch
from bioseq_models import load_model
from MotifSift import SynonymousCodonSpace, call_motifs, generate_designs, select_candidates
from saluki_support import ReporterScorer, codon_ism

device = "cuda" if torch.cuda.is_available() else "cpu"
cds = "".join(Path("data/egfp.fasta").read_text().splitlines()[1:])
saluki = load_model("saluki", device=device, crosses=(0,))
model = ReporterScorer(saluki, len(cds))
seeds = [cds]
print("Starting Saluki score:", float(model.score(seeds)[0, 0]))

run = Path(mkdtemp(prefix="motifsift-example-"))
```

## 2. Generate candidates

Run the eleven optimizers and keep 100 candidates per optimizer. The saved tables show the previous full run. New runs use the current MotifSift defaults.

```python
designs = generate_designs(
    model, length=len(cds), initial=seeds, objective="maximize",
    gain_fractions=(0.5, 0.9), output_dir=run / "designs",
    is_CDS_optimized=True, fixed_region=[slice(len(cds) - 3, len(cds))],
)
designs.groupby(["algorithm", "gain_fraction"])["score"].agg(["size", "median", "max"])
```

## 3. Discover motifs

Analyze the 90% gain stage with synonymous-codon attribution and forward-strand motif discovery.

```python
designs = designs.loc[designs["gain_fraction"].eq(0.9)].copy()
space = SynonymousCodonSpace(cds).with_fixed_positions(cds, range(len(cds) - 3, len(cds)))
contributions = codon_ism(model, designs.drop_duplicates("sequence"), space)
motifs = call_motifs(
    designs, reverse_complement=False, contributions=contributions,
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
