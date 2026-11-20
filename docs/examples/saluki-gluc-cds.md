# Saluki for synonymous Gluc CDS design

[Notebook](../../examples/saluki-gluc-cds.ipynb) · [Installation](../installation.md)

Optimize the Gluc CDS for predicted mRNA stability. Keep translation and both UTRs fixed.

## 1. Load the model

```python
from pathlib import Path

import torch
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, select_candidates

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cds = (
    "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCTGTGGCCGAGGCCAAGCCCACCGAGAACAACGAA"
    "GACTTCAACATCGTGGCCGTGGCCAGCAACTTCGCGACCACGGATCTCGATGCTGACCGCGGGAAGTTGCCC"
    "GGCAAGAAGCTGCCGCTGGAGGTGCTCAAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGC"
    "TGTCTGATCTGCCTGTCCCACATCAAGTGCACGCCCAAGATGAAGAAGTTCATCCCAGGACGCTGCCACACC"
    "TACGAAGGCGACAAAGAGTCCGCACAGGGCGGCATAGGCGAGGCGATCGTCGACATTCCTGAGATTCCTGGG"
    "TTCAAGGACTTGGAGCCCATGGAGCAGTTCATCGCACAGGTCGATCTGTGTGTGGACTGCACAACTGGCTGC"
    "CTCAAAGGGCTTGCCAACGTGCAGTGTTCTGACCTGCTCAAGAAGTGGCTGCCGCAACGCTGTGCGACCTTT"
    "GCCAGCAAGATCCAGGGCCAGGTGGACAAGATCAAGGGGGCCGGTGGTGACTAA"
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
        return saluki.score_soft(full_transcripts.clamp(0, 1))


model = CDSScorer()
seeds = [cds]

run = Path("outputs/saluki_gluc_cds")
```

## 2. Generate candidates

```python
designs = generate_designs(
    model, length=len(cds), initial=seeds, objective="maximize",
    gain_fractions=(0.5, 0.9), output_dir=run / "designs",
    is_CDS_optimized=True, fixed_region=[slice(len(cds) - 3, len(cds))],
)
designs.groupby(["algorithm", "gain_fraction"])["score"].agg(["size", "median", "max"])
```

## 3. Discover motifs

Use the 90% gain stage for motif discovery and panel selection.

```python
designs = designs.loc[designs["gain_fraction"].eq(0.9)].copy()
motifs = call_motifs(designs, model, reverse_complement=False, output_dir=run / "motifs")
motifs.support.query("retained")[["motif_id", "seqlets", "parents", "sites_on_unique_DNA"]]
```

## 4. Select a panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.2))
selected = select_candidates(designs, motifs=motifs, top_n=10, min_score=gate, output_dir=run / "selected")
selected[["selection_rank", "sequence", "score", "present_motifs"]]
```
