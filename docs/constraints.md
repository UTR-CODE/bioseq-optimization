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

## Hard GC and motif constraints

```python
import numpy as np
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize, random_seed_sequences


def gc_at_most_60(sequences):
    return np.asarray(
        [
            (sequence.count("G") + sequence.count("C")) / len(sequence) <= 0.60
            for sequence in sequences
        ],
        dtype=float,
    )


def excludes_motifs(sequence):
    unwanted = ("AAAA", "GGGG", "GAATTC")
    return float(all(motif not in sequence for motif in unwanted))


predictor = load_model("optimus5prime", device="cpu")
initial = random_seed_sequences(length=50, count=4, seed=7)
result = genetic_algorithm_optimize(
    predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    constraint=[gc_at_most_60, excludes_motifs],
    budget=1000,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
print(result.to_frame())
print(result.metadata["constraint"])
```

`gc_at_most_60` handles a batch. `excludes_motifs` handles one string; the
optimizer calls a single-sequence function once per candidate. Boolean values
are accepted because they convert to zero and one.

For example, if the GC constraint passes 800 of 1,000 screened candidates and
the motif constraint passes 600, only candidates present in both passing sets
reach the predictor. The exact intersection size is reported separately; it
need not be 480 because the two constraints can be correlated.

## Soft GC constraint

This function always passes sequences in the 40–60% interval. Outside that
interval, the pass probability decreases linearly with distance:

```python
def gc_between_40_and_60(sequences):
    probabilities = []
    for sequence in sequences:
        gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
        if 0.40 <= gc <= 0.60:
            probability = 1.0
        elif gc < 0.40:
            probability = max(0.0, gc / 0.40)
        else:
            probability = max(0.0, (1.0 - gc) / 0.40)
        probabilities.append(probability)
    return probabilities


result = genetic_algorithm_optimize(
    predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    constraint=gc_between_40_and_60,
    budget=1000,
    seed=7,
)
```

For each unique candidate, the optimizer draws once from each returned pass
probability. Draws from different constraints are independent and reproducible
with `seed`. A repeated sequence keeps its first screening outcome.

## Results and query counts

`result.constraint_scores` has shape `[num_candidates, num_constraints]` and
contains the probabilities returned for the retained sequences.
`result.constraint_names` identifies the columns. `result.to_frame()` adds one
column per callable, such as `constraint_gc_at_most_60`.

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
