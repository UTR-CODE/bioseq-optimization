# Custom sequence constraints

[Notebook](../../examples/custom-constraints.ipynb)

Pass a function to `constraint=` that receives a batch of sequences and returns one Boolean per sequence. `True` allows the sequence. The constraint applies to seeds and proposed candidates. To exclude a pattern on both DNA strands, include its reverse complement.

## Define the filters

```python
import re

from MotifSift import generate_designs, random_seed_sequences

forbidden = ("GGTACC", "GGATCC")
homopolymer = re.compile(r"([ACGT])\1{5}")

def gc_fraction(sequence):
    return (sequence.count("G") + sequence.count("C")) / len(sequence)

def constraint(sequences):
    return [
        0.35 <= gc_fraction(sequence) <= 0.65
        and not any(pattern in sequence for pattern in forbidden)
        and homopolymer.search(sequence) is None
        for sequence in sequences
    ]

constraint(["AGTACGATACGT", "AAGGATCCTTAC", "AAAATATATTAT", "GGGGGGATATAT"])
```

## Optimize with the constraint

```python
class GCountScorer:
    def score(self, sequences):
        return [sequence.count("G") for sequence in sequences]

    def score_soft(self, probabilities):
        return probabilities[:, :, 2].sum(dim=1)

length = 24
seeds = random_seed_sequences(length, 8)
designs = generate_designs(
    GCountScorer(), length=length, initial=seeds, objective="maximize",
    num_candidates=8, score_threshold=10, constraint=constraint,
)
designs.assign(gc_fraction=designs.sequence.map(gc_fraction))[
    ["sequence", "algorithm", "score", "gc_fraction"]
].head()
```

## Keep selected positions fixed

`fixed_region` uses zero-based positions and Python slices. In `slice(0, 4)`, the index 4 is excluded. Every starting sequence must already agree at the fixed positions. This example keeps the first four bases and the last three bases from a shared template.

```python
prefix, suffix = "ACGT", "TAC"
initial = [prefix + seed[len(prefix):-len(suffix)] + suffix for seed in seeds]
fixed_designs = generate_designs(
    GCountScorer(), length=length, initial=initial, objective="maximize",
    num_candidates=8, score_threshold=10,
    fixed_region=[slice(0, len(prefix)), slice(-len(suffix), None)],
)
fixed_designs[["algorithm", "sequence", "score"]].head()
```
