# Saluki for synonymous eGFP CDS design

[Notebook](../../examples/saluki-egfp.ipynb) · [Installation](../installation.md)

This example uses Saluki to score a single-exon eGFP transcript. The UTRs are fixed and the search only introduces synonymous changes to the CDS.

## Load eGFP and Saluki

```python
from pathlib import Path

import torch
from bioseq_models import load_model
from sequence_optimization import CallableScorer, generate_designs

cds = "".join(Path("examples/data/egfp.fasta").read_text().splitlines()[1:])
utr5 = "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACC"
utr3 = "GCTGGAGCCTCGGTGGCCTAGCTTCTTGCCCCTTGGGCCTCCCCCCAGCCCCTCCTCCCCTTCCTGCACCCGTACCCCCGTGGTCTTTGAATAAAGTCTGAGTGGGCGGCA"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(
    "saluki", device=device, cds_start=len(utr5), cds_end=len(utr5) + len(cds),
    exon_lengths=[len(utr5) + len(cds) + len(utr3)],
)

def score_cds(sequences):
    return model.score([utr5 + seq + utr3 for seq in sequences])

scorer = CallableScorer(score_cds, task_names=("half_life_score",))
```

## Optimize the CDS and compare scores

```python
designs = generate_designs(
    scorer, length=len(cds), initial=[cds], objective=["maximize"],
    is_CDS_optimized=True, fixed_region=[slice(len(cds) - 3, len(cds))],
    algorithms=["genetic_algorithm"], num_candidates=4, seed=42, parallel=False,
)
print("Starting Saluki score:", float(score_cds([cds])[0, 0]))
designs[["sequence", "activity_half_life_score"]]
```
