# Search workflow

The five required optimizer arguments are `model`, `length`, `initial`,
`objective`, and `num_candidates`. Other arguments have defaults. A search can
start from supplied sequences, reproducible random sequences, or sequences
loaded from a text file.

## Starting sequences

When `initial` is supplied, it must contain exactly `num_candidates` unique
sequences. Every sequence must use the requested alphabet and have the
requested length. The optimizer returns that same number of unique retained
candidates.

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize, random_seed_sequences

model = load_model("optimus5prime", device="cpu")

initial = random_seed_sequences(length=50, count=4, seed=7)

result = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    budget=1000,
    seed=7,
)
```

Pass `initial=None` to let an optimizer sample its own random starting set. To
share the same starting set between runs, call `random_seed_sequences` in each
example with the same arguments:

```python
from sequence_optimization import random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
```

The helper returns exactly `count` unique sequences when the sequence space is
large enough. The same length, alphabet, count, and seed reproduce the same
array.

To load a fixed set from a text file, put one sequence on each non-empty line
and use `load_seed_sequences`:

```python
from sequence_optimization import load_seed_sequences

initial = load_seed_sequences(
    "optimus_seeds.txt",
    length=50,
    count=4,
)
```

Blank lines and lines whose first non-whitespace character is `#` are ignored.
The loader normalizes lowercase DNA and RNA notation, validates sequence length
and alphabet, rejects duplicates, and checks `count` when it is provided.

For a protein-preserving CDS search, pass complete CDSs through `initial` and
set `is_CDS_optimized=True` on `genetic_algorithm_optimize`. Each input is
validated and optimized independently. One input returns one result; multiple
inputs return a result list in the same order. See
[synonymous CDS optimization](tasks/synonymous_cds.md).

## Fixed regions

Pass a list through `fixed_region` to prevent selected positions from changing.
Entries may be integer positions, Python `slice` objects, or `"start:stop"`
strings. Coordinates are zero-based, the start is included, the stop is
excluded, and negative values count from the end.

```python
result = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    fixed_region=["1:3", "-3:-1", "-35:-20"],
    budget=1000,
    seed=7,
)
```

For a 50-base sequence, this fixes positions 1–2, 47–48, and 15–29. The
equivalent Python form is
`fixed_region=[slice(1, 3), slice(-3, -1), slice(-35, -20)]`. Use `[0]` or
`["0:1"]` for the first base and `[-1]` or `["-1:"]` for the final base.
Overlapping entries are combined.

With explicit non-CDS `initial` sequences, the fixed bases come from the first
sequence and every other initial sequence must agree at those positions. With
`initial=None`, one shared fixed template is drawn reproducibly from `seed`. Fixed
positions apply to random proposals, mutations, crossover, generated sequences,
and differentiable optimization. In synonymous CDS optimization, codon choices
are restricted to preserve both the fixed bases and the encoded protein.

The maximum possible pairwise Hamming distance is the number of mutable
positions. If `diversity` exceeds that number while multiple results are
requested, the optimizer raises `ValueError`.

## Candidate constraints

Use the `constraint` parameter to screen candidates by GC content, motifs, or
other sequence rules before model prediction. It accepts one callable or a
list whose independent passing sets are intersected. Hard constraints return
zero or one; values between zero and one define reproducible pass
probabilities. See [sequence constraints](constraints.md) for complete examples,
result columns, and screening counts.

## Candidate count and budgets

`num_candidates` controls both the number of starting sequences and the number
of sequences retained in the result. In fixed-budget mode, `budget` defaults to
10,000 and must be at least `num_candidates`. It limits candidate query or
proposal accounting; it is not a training epoch count and does not mean that
the final result contains that many sequences.

Discrete optimizers use a cache. With the default `count_repeated=False`, a
sequence already scored by the wrapped scorer is a cache hit and does not
consume another counted query. Set `count_repeated=True` when every proposal,
including repeated sequences, should consume budget. The result's
`query_count` reports the counted amount.

The gradient optimizers (`vanilla_activation_maximization_optimize`,
`fast_seqprop_optimize`, `den_optimize`, and `ledidi_optimize`) require a
scorer with `score_soft`. Their additional soft forward and backward passes do
not appear in the discrete `query_count`. These optimizers count repeated
discrete proposals. CbAS also counts repeated proposals because it trains an
internal proposal generator, even though it uses discrete scoring. Use a
`DifferentiableSequencePredictor` with the model's encoder and input layout for
the gradient methods; see the [Optimus example](models/optimus.md).

When several independent models are combined with `CombinedScorer`, the
optimizer budget remains the candidate/proposal budget for that run. Each
uncached candidate is sent to every child, so a three-child scorer can perform
three child-model sequence evaluations for one counted candidate. The
`model_query_counts` and `model_soft_query_counts` dictionaries are cumulative
for the scorer lifetime, not per-run; take deltas when comparing runs. The
optimizer cache determines which sequences reach the children. A child that
contains an internal checkpoint ensemble still contributes one child counter
entry per sequence at the scorer boundary; its internal folds or
forward/reverse passes are not additional `CombinedScorer` children or extra
optimizer-budget candidates.

## Stop at saturation

Pass `saturation=SaturationConfig(...)` to an optimizer to continue until the
retained-candidate mean score stops making material progress, then return
candidate sets at selected fractions of the observed gain. `maximum_queries`
is a safety ceiling.

When `saturation` is set, omit `budget`; supplying both raises `ValueError`.
When `saturation=None`, omitting `budget` uses 10,000. Saturation query limits are:

| Parameter | Default | Meaning |
|---|---|---|
| `fractions` | `(0.8, 0.9)` | Fractions of observed baseline-to-plateau gain to return |
| `minimum_queries` | `10 * num_candidates` | Minimum counted queries before plateau stopping is allowed |
| `plateau_patience_queries` | `20 * num_candidates` | Counted queries without material improvement before plateau stopping |
| `maximum_queries` | `1000 * num_candidates` | Safety ceiling on counted queries |
| `material_improvement` | `1e-4` | Minimum top-candidate-mean improvement considered material |

```python
from sequence_optimization import (
    SaturationConfig,
    genetic_algorithm_optimize,
    random_seed_sequences,
)

initial = random_seed_sequences(length=50, count=4, seed=7)
saturated = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9),
        minimum_queries=40,
        plateau_patience_queries=80,
        maximum_queries=400,
    ),
    seed=7,
)

q80 = saturated[0.8]
q90 = saturated.at(0.9)
print(saturated.stop_reason)
print(q80.metadata["stop_reason"])
print(q90.metadata["stop_reason"])
print(saturated.total_query_count)
```

The selection fraction is a fraction of gain, not a percentile of candidate
scores. If `baseline` is the top-candidate mean in the first complete archive
and `plateau` is the final observed top-candidate mean, q80 is the earliest
archive whose top-candidate mean reaches
`baseline + 0.8 * (plateau - baseline)`. q90 uses `0.9` in the same formula.
The run must finish its plateau or safety-ceiling procedure before q80 and q90
can be selected; requesting a lower fraction does not stop the run early or
reduce the queries needed to establish the observed plateau. These selections
also describe the observed search trajectory, not a guarantee of a global
optimum.
The returned `SaturationResult` exposes `stop_reason`, `total_query_count`,
`knee_query_count`, and `trajectory`; each q-result stores the stop reason and
the baseline and plateau means in `metadata`.

A stop reason of `"plateau"` indicates that the configured patience window
found no material improvement. `"maximum_queries"` indicates that the safety
ceiling was reached, and `"algorithm_stopped"` indicates that the selected
algorithm ended before either condition. Saturation stopping is an observed
progress rule with a safety ceiling; it does not guarantee that a biological
or model-specific optimum has been reached.

## Objectives and returned data

Pass one entry per model output: `"maximize"`, `"minimize"`, `"ignore"`, or
a finite numeric target. A numeric entry is a raw-output lower target. Once it
is reached, later selection preserves that threshold while optimizing any
remaining directions; a target-only search stops when one candidate satisfies
all numeric entries. Numeric targets use fixed-budget mode and cannot be
combined with saturation. Active outputs use an equal-weight signed mean. Use
[`MultiTaskObjective`](objectives.md) for fixed centers, scales, another
aggregation, or explicit lower and upper bounds.

An `OptimizationResult` provides `sequences`, scalar `scores`, raw model
`activities`, `uncertainties`, `task_names`, `query_count`, `metadata`, and an
optional `trajectory`. `result.to_frame()` presents sequence, scalar score,
uncertainty, attached activity columns, constraint-probability columns, and
objective-evaluation margin or satisfaction columns when available. It does
not include arbitrary metadata or the trajectory; read those from
`result.metadata` and `result.trajectory`.
