# Motif discovery

[Home](../README.md) · [Motif-diversity selection](motif-diversity.md)

The workflow extracts high-contribution seqlets, discovers sequence patterns, and builds a motif catalog for candidate selection.

```text
Unique candidates → attribution → positive seqlets
                  → motif discovery → support filtering → PWM catalog
```

The following steps continue from the [enhancer example](examples/enhancer.md), using its `scorer`, `objective`, and `candidates`.

## 1. Calculate nucleotide contributions

```python
from pathlib import Path
import numpy as np
import pandas as pd
from sequence_optimization import ism_contributions
from sequence_optimization.workflows import deduplicate_candidates, write_fasta

WORK = Path("outputs/motifs")
WORK.mkdir(parents=True, exist_ok=True)
unique, provenance = deduplicate_candidates(candidates)

def fitness(sequences):
    activities = scorer.score(sequences).scores
    return objective.scalarize(activities, scorer.task_names)

contribution = ism_contributions(unique.sequence.to_numpy(), fitness, batch_size=512)
np.savez_compressed(WORK / "contributions.npz", observed=contribution.observed,
                    sequence_ids=unique.sequence_id.to_numpy(dtype=str))
```

ISM scores all three alternative bases at every position. The observed-base contribution is the original prediction minus the four-base mean. DeepLIFTShap can supply the same `[sequences,length]` projected contribution matrix using dinucleotide-shuffled references.

## 2. Extract positive seqlets

```python
import torch
from tangermeme.seqlet import recursive_seqlets
seqlets = recursive_seqlets(
    torch.tensor(contribution.observed, dtype=torch.float32),
    threshold=0.01, min_seqlet_len=6, max_seqlet_len=25, additional_flanks=2,
)
seqlets["example_idx"] = seqlets.example_idx.astype(int)
seqlets["start"] = seqlets.start.astype(int).clip(lower=0)
seqlets["end"] = seqlets.end.astype(int).clip(upper=len(unique.sequence.iloc[0]))
seqlets["sequence_id"] = seqlets.example_idx.map(unique.sequence_id.to_dict())
seqlets["dna"] = [unique.sequence.iloc[r.example_idx][r.start:r.end]
                   for r in seqlets.itertuples()]
seqlets = seqlets[seqlets.dna.str.len().ge(6) & seqlets.attribution.gt(0)].copy()
seqlets["seqlet_id"] = [f"Q{i+1:06}" for i in range(len(seqlets))]
seqlets.to_csv(WORK / "seqlets.csv", index=False)
write_fasta(WORK / "seqlets.fa", seqlets.seqlet_id, seqlets.dna)
```

The seqlet table retains each fragment's parent sequence and coordinates. See the [tangermeme seqlet interface](https://tangermeme.readthedocs.io/en/latest/api/seqlet.html).

## 3. Create composition-matched backgrounds

```python
from tangermeme.ersatz import dinucleotide_shuffle
from sequence_optimization import onehot
controls = {}
for length, group in seqlets.groupby(seqlets.dna.str.len()):
    x = torch.tensor(onehot(group.dna.to_numpy(), channels_first=True))
    shuffled = dinucleotide_shuffle(x, n=1, random_state=42)
    codes = shuffled[:, 0].argmax(dim=1).cpu().numpy()
    for sid, row in zip(group.seqlet_id, codes):
        controls[sid] = "".join(np.array(list("ACGT"))[row])
write_fasta(WORK / "background.fa", seqlets.seqlet_id,
            [controls[sid] for sid in seqlets.seqlet_id])
```

## 4. Discover candidate motifs with STREME

```bash
streme --dna --p outputs/motifs/seqlets.fa --n outputs/motifs/background.fa \
  --oc outputs/motifs/streme --minw 6 --maxw 15 --nmotifs 30 --seed 42
```

For independent enrichment evaluation, partition parent sequence families into discovery and holdout sets before this step. Discover motifs on the discovery seqlets, then evaluate motif presence against matched backgrounds in held-out families.

## 5. Retain supported motifs

Scan the discovered PWMs back onto seqlets and retain motifs supported by at least ten seqlets and three parent sequences.

```python
from sequence_optimization.motifs import read_meme, write_meme, scan_fimo
pwms = read_meme(WORK / "streme/streme.txt")
hits = scan_fimo(WORK / "streme/streme.txt", WORK / "seqlets.fa", WORK / "seqlet_scan")
rows = []
for motif_id in pwms:
    ids = set(hits.loc[hits.motif_id.eq(motif_id), "sequence_name"])
    supported = seqlets[seqlets.seqlet_id.isin(ids)]
    rows.append({"motif_id": motif_id, "seqlets": len(supported),
                 "parents": supported.sequence_id.nunique()})
support = pd.DataFrame(rows)
support.to_csv(WORK / "motif_support.csv", index=False)
retained = support.loc[(support.seqlets >= 10) & (support.parents >= 3), "motif_id"]
write_meme(WORK / "catalog.meme", {m: pwms[m] for m in retained})
```

The subsequent [candidate scan](motif-diversity.md) retains motifs with at least 30 physical sites across the source candidate records.

## 6. Catalog organization and annotation

For catalog-level redundancy analysis, use [Almeida motif-clustering](https://github.com/bernardo-de-almeida/motif-clustering). Its clustering uses Tomtom similarity profiles and complete linkage. The repository includes an adapter for the upstream R core.

```bash
git clone https://github.com/bernardo-de-almeida/motif-clustering.git external/motif-clustering
tomtom -oc outputs/motifs/tomtom -dist kullback -motif-pseudo 0.1 \
  -min-overlap 1 -thresh 0.5 outputs/motifs/catalog.meme outputs/motifs/catalog.meme
```

```python
tomtom = pd.read_csv(WORK / "tomtom/tomtom.tsv", sep="\t", comment="#").dropna(subset=["Query_ID", "Target_ID"])
ids = list(read_meme(WORK / "catalog.meme"))
evalues = pd.DataFrame(100., index=ids, columns=ids)
for _, row in tomtom.iterrows():
    evalues.loc[row["Query_ID"], row["Target_ID"]] = float(row["E-value"])
evalues.to_csv(WORK / "tomtom_evalues.csv")
```

```bash
Rscript tools/run_almeida_motif_dedup.R outputs/motifs/tomtom_evalues.csv \
  outputs/motifs 0.8 external/motif-clustering/Motif_clustering.R
```

Inspect motif families and retain representative PWMs with their support metadata. The sequence-selection stage uses continuous PWM similarity, preserving informative flank differences between related motifs. Annotate catalog PWMs against a species-matched motif database with Tomtom.
