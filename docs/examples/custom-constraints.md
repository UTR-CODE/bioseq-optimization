# Custom sequence constraints

[Notebook](../../examples/custom-constraints.ipynb)

Pass a function to `constraint=` that receives a batch of sequences and returns one Boolean per sequence. `True` allows the sequence. MotifSift applies the filter to starting sequences and proposed candidates.

## Define the filters

```python
import re

import pandas as pd
from MotifSift import generate_designs, random_seed_sequences

forbidden_patterns = ("GGTACC", "GGATCC")
long_homopolymer = re.compile(r"([ACGT])\1{5}")


def gc_fraction(sequence):
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def avoid_patterns(sequences):
    return [not any(pattern in sequence for pattern in forbidden_patterns) for sequence in sequences]


def limit_gc(sequences):
    return [0.35 <= gc_fraction(sequence) <= 0.65 for sequence in sequences]


def limit_homopolymers(sequences):
    return [long_homopolymer.search(sequence) is None for sequence in sequences]


cases = ["AGTACGATACGT", "AAGGATCCTTAC", "AAAATATATTAT", "GGGGGGATATAT"]
pd.DataFrame({
    "sequence": cases,
    "allowed patterns": avoid_patterns(cases),
    "35–65% GC": limit_gc(cases),
    "no run of six": limit_homopolymers(cases),
})
```

`avoid_patterns` checks exact matches in the supplied sequence. To exclude a non-palindromic pattern on both DNA strands, include its reverse complement in `forbidden_patterns`.

## Combine filters and optimize

Pass any filter on its own, or require all three to pass. The example uses a scorer that counts G bases so the constraint logic is easy to verify.

```python
def constraint(sequences):
    return [all(results) for results in zip(
        avoid_patterns(sequences), limit_gc(sequences), limit_homopolymers(sequences),
    )]


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
assert all(constraint(designs.sequence.tolist()))

designs.assign(gc_fraction=designs.sequence.map(gc_fraction))[[
    "sequence", "algorithm", "score", "gc_fraction",
]].head()
```
