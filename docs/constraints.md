# Sequence constraints

Every optimizer accepts `constraint`. Supply one callable or a list of
callables. Each callable receives a sequence or a NumPy array of sequences and
returns one value per sequence in `[0, 1]`:

- `1` always passes the constraint.
- `0` always fails the constraint.
- A value between zero and one is the probability of passing that constraint.

Constraints screen candidates before model prediction. When several
constraints are supplied, each runs independently and only candidates passing
every constraint—the intersection—are sent to the model and admitted to the
next optimization round.

## Copyable hard constraints

Each factory below returns a batch constraint that can be passed to any
optimizer. Copy the functions needed for a design and adjust their arguments.

```python
import numpy as np


def gc_between(minimum=0.40, maximum=0.60):
    def constraint(sequences):
        fractions = np.asarray([
            (sequence.count("G") + sequence.count("C")) / len(sequence)
            for sequence in sequences
        ])
        return ((minimum <= fractions) & (fractions <= maximum)).astype(float)

    constraint.__name__ = "gc_between"
    return constraint


def excludes_motifs(*motifs):
    motifs = tuple(motif.upper().replace("U", "T") for motif in motifs)

    def constraint(sequences):
        return np.asarray([
            all(motif not in sequence for motif in motifs)
            for sequence in sequences
        ], dtype=float)

    constraint.__name__ = "excludes_motifs"
    return constraint


def requires_all_motifs(*motifs):
    motifs = tuple(motif.upper().replace("U", "T") for motif in motifs)

    def constraint(sequences):
        return np.asarray([
            all(motif in sequence for motif in motifs)
            for sequence in sequences
        ], dtype=float)

    constraint.__name__ = "requires_all_motifs"
    return constraint


def homopolymer_at_most(maximum_run=4):
    forbidden = tuple(base * (maximum_run + 1) for base in "ACGT")
    constraint = excludes_motifs(*forbidden)
    constraint.__name__ = "homopolymer_at_most"
    return constraint


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def avoids_restriction_sites(*sites):
    patterns = {
        pattern
        for site in sites
        for normalized in (site.upper().replace("U", "T"),)
        for pattern in (normalized, reverse_complement(normalized))
    }
    constraint = excludes_motifs(*sorted(patterns))
    constraint.__name__ = "avoids_restriction_sites"
    return constraint


def has_flanks(prefix="", suffix=""):
    prefix = prefix.upper().replace("U", "T")
    suffix = suffix.upper().replace("U", "T")

    def constraint(sequences):
        return np.asarray([
            sequence.startswith(prefix) and sequence.endswith(suffix)
            for sequence in sequences
        ], dtype=float)

    constraint.__name__ = "has_flanks"
    return constraint
```

For example, these constraints retain 40–60% GC sequences, reject five-base
homopolymers, and exclude EcoRI, BsaI, and BsmBI sites on either strand:

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize, random_seed_sequences


predictor = load_model("optimus5prime", device="cpu")
initial = random_seed_sequences(length=50, count=4, seed=7)
result = genetic_algorithm_optimize(
    predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    constraint=[
        gc_between(0.40, 0.60),
        homopolymer_at_most(4),
        avoids_restriction_sites("GAATTC", "GGTCTC", "CGTCTC"),
    ],
)
print(result.to_frame())
print(result.metadata["constraint"])
```

The optimizer also accepts a function that handles one sequence at a time. For
example, `lambda sequence: float("TATAAA" not in sequence)` excludes one motif.
Boolean values are accepted because they convert to zero and one.

For example, if the GC constraint passes 800 of 1,000 screened candidates and
the motif constraint passes 600, only candidates present in both passing sets
reach the predictor. The exact intersection size is reported separately; it
need not be 480 because the two constraints can be correlated.

## Copyable soft GC constraint

This function always passes sequences in the 40–60% interval. Outside that
interval, the pass probability decreases linearly with distance:

```python
def soft_gc_between(minimum=0.40, maximum=0.60):
    def constraint(sequences):
        probabilities = []
        for sequence in sequences:
            gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
            if minimum <= gc <= maximum:
                probability = 1.0
            elif gc < minimum:
                probability = max(0.0, gc / minimum)
            else:
                probability = max(0.0, (1.0 - gc) / (1.0 - maximum))
            probabilities.append(probability)
        return probabilities

    constraint.__name__ = "soft_gc_between"
    return constraint


result = genetic_algorithm_optimize(
    predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    constraint=soft_gc_between(0.40, 0.60),
)
```

For each unique candidate, the optimizer draws once from each returned pass
probability. Draws from different constraints are independent and reproducible
with `seed`. A repeated sequence keeps its first screening outcome.

## Results and query counts

`result.constraint_scores` has shape `[num_candidates, num_constraints]` and
contains the probabilities returned for the retained sequences.
`result.constraint_names` identifies the columns. `result.to_frame()` adds one
column per callable, using a numeric suffix when callable names repeat.

`result.metadata["constraint"]` contains each constraint's evaluated count,
passed count and pass rate, plus the intersection count and rate. Its
`screened_query_count` is the number of unique sequences evaluated by the
constraints, and `model_query_count` is the number that passed the complete
intersection and reached discrete model scoring. `result.query_count`, also
recorded as `counted_query_count`, follows the optimizer's configured
budget-counting mode. Preparing enough valid starting candidates can evaluate
additional sequences before the query budget begins.

Constraint functions must return a finite one-dimensional value array with one
entry per sequence. Values outside `[0, 1]`, a wrong number of values, or an
empty constraint list raise an error. If the constraints cannot admit
`num_candidates` unique starting sequences, the optimizer raises an error.

For gradient optimizers, arbitrary Python constraints cannot be differentiated.
They screen every discrete sequence sampled during optimization; only passing
samples contribute to the differentiable model objective, the next update, and
the retained archive.

Synonymous-CDS structural rules remain active before user constraints. See
[synonymous CDS constraints](tasks/synonymous_cds.md).
