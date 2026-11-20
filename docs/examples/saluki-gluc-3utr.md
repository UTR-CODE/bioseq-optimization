# Saluki for Gluc 3′ UTR design

[Notebook](../../examples/saluki-gluc-3utr.ipynb) · [Installation](../installation.md)

Optimize a 150-nt Gluc 3′ UTR for predicted mRNA stability. Keep the HBB 5′ UTR and Gluc CDS fixed.

## 1. Load the model

```python
from pathlib import Path

import torch
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, random_seed_sequences, select_candidates

utr5 = "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACC"
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
length = 150
prefix = utr5 + cds

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
saluki = load_model(
    "saluki", device=device,
    cds_start=len(utr5), cds_end=len(prefix),
    exon_lengths=[len(prefix) + length],
)

indices = torch.tensor(["ACGT".index(base) for base in prefix])
fixed = torch.nn.functional.one_hot(indices, num_classes=4)


class UTRScorer:
    task_names = saluki.task_names
    device = saluki.device

    def score(self, sequences):
        return saluki.score([prefix + sequence for sequence in sequences])

    def score_soft(self, probabilities):
        fixed_batch = fixed.to(probabilities).expand(len(probabilities), -1, -1)
        transcripts = torch.cat([fixed_batch, probabilities], dim=1)
        return saluki.score_soft(transcripts)


model = UTRScorer()
run = Path("outputs/saluki_gluc_3utr")
```

## 2. Generate candidates

```python
seeds = random_seed_sequences(length, 100, seed=42)
designs = generate_designs(
    model, length=length, initial=seeds, objective="maximize",
    output_dir=run / "designs",
)
designs.groupby("algorithm")["score"].agg(["size", "median", "max"])
```

## 3. Discover motifs

```python
motifs = call_motifs(designs, model, reverse_complement=False, output_dir=run / "motifs")
motifs.support.query("retained")[["motif_id", "seqlets", "parents", "sites_on_unique_DNA"]]
```

## 4. Select a panel

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.2))
selected = select_candidates(designs, motifs=motifs, top_n=10, min_score=gate, output_dir=run / "selected")
selected[["selection_rank", "sequence", "score", "present_motifs"]]
```
