# Saluki for synonymous eGFP CDS design

[Notebook](../../examples/saluki-egfp.ipynb) · [Installation](../installation.md)

Score a single-exon eGFP transcript with fixed UTRs. Optimize the CDS with synonymous substitutions and retain the terminal stop codon. Install MEME Suite with the notebook dependencies. The first model load downloads the pretrained checkpoint.

## 1. Load the model

```python
import os
from pathlib import Path

import torch
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, select_candidates

device = "cuda" if torch.cuda.is_available() else "cpu"
cds = (
    "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAAC"
    "GGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATC"
    "TGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTC"
    "AGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAG"
    "CGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTG"
    "GTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTAC"
    "AACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATC"
    "CGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGC"
    "CCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGC"
    "GATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
)
utr5 = "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACC"
utr3 = "GCTGGAGCCTCGGTGGCCTAGCTTCTTGCCCCTTGGGCCTCCCCCCAGCCCCTCCTCCCCTTCCTGCACCCGTACCCCCGTGGTCTTTGAATAAAGTCTGAGTGGGCGGCA"
saluki = load_model(
    "saluki", device=device,
    cds_start=len(utr5), cds_end=len(utr5) + len(cds),
    exon_lengths=[len(utr5) + len(cds) + len(utr3)],
)


def encode_utr(sequence):
    indices = torch.tensor(["ACGT".index(base) for base in sequence])
    return torch.nn.functional.one_hot(indices, num_classes=4)


left, right = encode_utr(utr5), encode_utr(utr3)


class CDSScorer:
    task_names = saluki.task_names
    device = saluki.device

    def score(self, sequences):
        return saluki.score([utr5 + sequence + utr3 for sequence in sequences])

    def score_soft(self, probabilities):
        batch = len(probabilities)
        full_transcripts = torch.cat([
            left.to(probabilities).expand(batch, -1, -1),
            probabilities,
            right.to(probabilities).expand(batch, -1, -1),
        ], dim=1)
        return saluki.score_soft(full_transcripts)


model = CDSScorer()
seeds = [cds]
print("Starting Saluki score:", float(model.score(seeds)[0, 0]))

run = Path("outputs/saluki_egfp_cds")
os.makedirs(run, exist_ok=True)
```

## 2. Generate candidates

Run the eleven optimizers and keep 100 candidates per optimizer. The saved design summary comes from the previous full search. The motif and panel tables come from reanalyzing its 90% gain candidates.

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
motifs = call_motifs(
    designs, model, reverse_complement=False,
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
