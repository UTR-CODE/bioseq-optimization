# Search workflow

The five required optimizer arguments are `model`, `length`, `initial`,
`objective`, and `num_candidates`. Other arguments have defaults. A search can
start from supplied sequences, reproducible random sequences, or sequences
loaded from a text file.

## Starting sequences

When `initial` is supplied, it must contain at least `num_candidates` unique
sequences. Every sequence must use the requested alphabet and have the
requested length. The optimizer scores the complete initial population and
returns `num_candidates` unique retained candidates.

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
set `is_CDS_optimized=True` on any optimizer. Each input is validated and
optimized independently. One input returns one result; multiple inputs return
a result list in the same order. See
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

## Candidate diversity

`diversity` is the minimum pairwise Hamming distance among the returned
sequences. The default is five; set it to zero to select candidates by score
alone. A larger value trades some objective score for candidates that differ at
more positions.

```python
initial = random_seed_sequences(length=50, count=4, seed=7)
result = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    diversity=10,
)
```

The maximum possible pairwise Hamming distance is the number of mutable
positions. If `diversity` exceeds that number while multiple results are
requested, the optimizer raises `ValueError`. If the scored archive does not
contain enough candidates at the requested distance, increase the budget or
lower `diversity`.

## Candidate constraints

Use the `constraint` parameter to screen candidates by GC content, motifs, or
other sequence rules before model prediction. It accepts one callable or a
list whose independent passing sets are intersected. Hard constraints return
zero or one; values between zero and one define reproducible pass
probabilities. See [sequence constraints](constraints.md) for complete examples,
result columns, and screening counts.

## Candidate count and budgets

`num_candidates` controls the number of sequences retained in the result. When
`initial=None`, it also controls the number of generated starting sequences. A
supplied initial population must be at least `num_candidates`; smaller
populations raise ValueError before optimization. For 100 returned candidates,
provide at least 100 unique starting sequences. A larger initial population is
allowed. CDS mode retains its per-reference generated-population behavior;
the number of reference CDSs is not the search population size.

The saturation baseline is the mean objective score of the initial scoring
batch, not the first archive containing `num_candidates` sequences. Before
enough diverse candidates exist, snapshots are ineligible for selection and
the search cannot declare a successful plateau. The initial mean/query count
anchors the elbow curve; only complete candidate sets can be returned. If a
90%-gain target was already crossed before the first complete set, that first
eligible set is returned. Metadata records `baseline_population_mean`,
`baseline_query_count`, and `first_complete_query_count`.
If no complete candidate set reaches the initial mean (e.g. a few unusually
strong seeds), return the best eligible complete snapshot and mark an unmet
90% threshold as `selection_target_reached=False`; never return fewer sequences
or claim a gain that was not achieved.

An algorithm stopping or exhausting its budget without enough candidates
raises a clear error; there is no duplicate padding or unrelated random
backfill. A collection phase without progress in available diverse count or
mean for the patience window also raises an error, avoiding endless gradient
iterations. Increase the initial population, exploration/patience or budget,
or reduce diversity when collection fails. Impossible sequence-space capacity
and empty/duplicate/invalid seeds are rejected before searching.
Every initial sequence consumes query budget, so `budget` must be at least the
initial population size. For every optimizer except random search,
the default `budget=None` places no fixed limit on candidate queries: the
optimizer stops from observed plateau progress and returns a quality-constrained,
diverse set around a late relaxed elbow.
Set `budget` to a positive integer for a fixed-budget run; it must be at least
`num_candidates`. A fixed budget limits candidate query or proposal accounting,
not training epochs or result size.

`random_optimize` always uses a fixed proposal budget and does not accept
`saturation`. Its default `budget=100_000` counts the initial candidates followed
by uniformly sampled random candidates. Repeated proposals consume this budget,
although the score cache prevents another model call for the same sequence.

Discrete optimizers use a cache. Default plateau searches count repeated
proposals so duplicate rounds advance the patience window without another model
call. In a fixed-budget or custom-saturation run with
`count_repeated=False`, a cached sequence does not consume another counted
query. Set `count_repeated=True` to count every proposal. The result's
`query_count` reports the counted amount.

CEM (`cross_entropy_method`) and simulated annealing always count repeated
proposals, even with `count_repeated=False`, just like gradient methods. A
random batch containing only cached sequences is not proof of convergence:
search continues until plateau, a target, or budget exhaustion. Cache hits
still avoid extra model inference. Thus their query counts include repeats
and are no longer directly comparable to old unique-query counts.

The gradient optimizers (`vanilla_activation_maximization_optimize`,
`fast_seqprop_optimize`, `den_optimize`, and `ledidi_optimize`) require a
scorer with `score_soft`. Their additional soft forward and backward passes do
not appear in the discrete `query_count`. These optimizers count repeated
discrete proposals. CbAS also counts repeated proposals because it trains an
internal proposal generator, even though it uses discrete scoring. Use a
`SequencePredictor` for a channels-first A/C/G/T PyTorch model. Subclass it for
another input representation; see [custom PyTorch models](custom_models.md).

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

With `budget` and `saturation` omitted, every optimizer except random search
continues until the retained-candidate mean score stops making material progress
and returns the default relaxed-knee result as an `OptimizationResult`.

Pass `saturation=SaturationConfig(...)` to customize the plateau rules or
request multiple gain fractions. This form returns a `SaturationResult`. Omit
`budget` whenever `saturation` is supplied; using both raises `ValueError`.
Random search does not accept `SaturationConfig`; use its fixed `budget`.
Saturation parameters are:

| Parameter | Default | Meaning |
|---|---|---|
| `fractions` | `(0.99,)` | Fractions of observed baseline-to-plateau gain to return |
| `minimum_queries` | `10 * N` | Minimum counted queries before plateau stopping is allowed |
| `plateau_patience_queries` | `max(40 * N, 3 * B)` | Counted queries without material improvement before plateau stopping |
| `maximum_queries` | `None` | Optional safety ceiling on counted queries |
| `material_improvement` | `1e-2` | Absolute floor for one-step material improvement |
| `relative_material_improvement` | `1e-3` | Relative threshold applied to the current score scale |
| `relaxed_knee` | `True` | Require the elbow to occur late in the run-relative gain curve |
| `knee_minimum_gain_fraction` | `0.85` | Minimum observed-gain fraction for a relaxed elbow candidate |
| `selection_minimum_gain_fraction` | `0.8` | Minimum selected-set mean, relative to observed gain |
| `knee_fraction_tolerance` | `0.1` | Half-width of the local knee band, relative to observed gain |
| `diversity_pool_size` | `20000` | Maximum sampled pool for diversity selection |

Here `N = max(num_candidates, len(initial))`; without explicit seeds it is
`num_candidates`. CDS runs use their generated population per reference, not
the number of reference CDSs. Explicit query limits are used unchanged.
`generate_designs()` uses the same defaults and has no implicit safety ceiling.
Random search remains fixed-budget. An explicit `maximum_queries` or fixed
`budget` is still honored.

Default selection is `selection="knee_fraction"` with `relaxed_knee=True`.
After the search stops, let `baseline` be the initial-population mean and let
`gain = peak_retained_mean - baseline`. Elbow candidates must reach at least
`baseline + 0.85 * gain`; among them, the maximum normalized distance above the
query/score chord is selected. This prevents a visually sharp but low-quality
early bend from defining the returned set.

The elbow mean itself is the quality target: the historical `knee_fraction=0.9`
discount is not applied a second time in relaxed mode. Candidate selection uses
the full evaluated archive, initially inside `target ± 0.1 * gain`, with a
selected-set mean floor of `baseline + 0.8 * gain`. Within this band, greedy selection
maximizes the sum of `log(1 + Hamming distance)` to already selected sequences;
score proximity only breaks ties. Every pair must still meet `diversity`.
Pools larger than `diversity_pool_size=20000` are reproducibly sampled.
If the local band is too small, it widens by score proximity. A diverse proposal
whose mean is initially below the 80% relative floor is blended with the
highest-mean diversity-feasible snapshot: the lowest-scoring diverse members
are replaced first, stopping after the minimum number of replacements that
meets the floor. A full fallback is used only if no feasible blend exists or it
would reduce the measured diversity. Metadata records the replacement count,
expansion, fallback, actual score range, and whether the mean floor was satisfied.
The selected sequences may come from after the elbow; selected query count
reports their latest discovery, not a reduced cost of running the search.
All thresholds above are relative to the scalarized objective trajectory of the
current run. No MRL, training-set, or other model-specific absolute cutoff is
used. Multi-task and minimization runs therefore use the same rule. Flat curves
fall back to the ordinary elbow logic because there is no material relative gain.

`SaturationResult.selected` exposes this default candidate set; `.knee` exposes
the elbow archive. Existing `result[0.9]` indexing still means a fraction of the
**final observed gain**, for compatibility. Use `selection="fraction"` to
explicitly retain that older selection policy, or `selection="knee"` for the
elbow itself. Metadata distinguishes the selection and stopping reasons.

The elbow is a retrospective selection, not an online stopping trigger.
Plateau stopping uses a dynamic one-step threshold:
`max(0.01, 0.001 * max(1, abs(current_top_mean)))`. Only a single observation
that clears this threshold refreshes patience. Consecutive smaller gains do not
accumulate into material progress, so a search that only creeps upward stops.
An algorithm-internal stop or explicit safety ceiling does not establish a
plateau. Without an upper limit a search can run for a long time.
Simulated annealing retains its existing nominal cooling horizon and temperature
floor; the nominal horizon is not a query ceiling.

For a gradient optimizer with N seeds, a full update proposes N sequences,
including repeated hard decodes. Thus `40 * N` allows approximately 40 updates
without material improvement. This is a counted-query window, not a universal
iteration count: CbAS/CEM batch sizes and ISM/beam neighborhood sizes differ.
`B` is the nominal proposal batch: N for population updates, the configured
batch size for BaseSearch/CbAS/CEM, or parents times single-mutant neighbors for
beam/ISM (beam parents use the configured elite fraction). The `3 * B` floor
prevents very large batches from making patience shorter than three full
proposal batches. Duplicate caching and finite remaining budget can make actual
batches smaller. A configured safety ceiling can still stop a run before this
window is exhausted. Resolved limits are recorded in selection metadata.
The minimum-query guard does not restart the patience clock; it only prevents
stopping before that query count. See [patience calibration](patience_calibration.md)
for the Optimus experiments and their limitations.

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
algorithm ended before either condition. `"target_reached"` indicates that a
target-only numeric objective was satisfied. Saturation stopping describes
observed progress; it does not guarantee a biological or model-specific optimum.

## Objectives and returned data

Pass one entry per model output: `"maximize"`, `"minimize"`, `"ignore"`, or
a finite numeric target. A numeric entry is a raw-output lower target. Once it
is reached, later selection preserves that threshold while optimizing any
remaining directions; a target-only search stops when one candidate satisfies
all numeric entries. Numeric targets also work with the default q99 mode and
with a custom `SaturationConfig`. Active outputs use an equal-weight signed mean. Use
[`MultiTaskObjective`](objectives.md) for fixed centers, scales, another
aggregation, or explicit lower and upper bounds.

An `OptimizationResult` provides `sequences`, scalar `scores`, raw model
`activities`, `uncertainties`, `task_names`, `query_count`, `metadata`, and an
optional `trajectory`. `result.to_frame()` presents sequence, scalar score,
uncertainty, attached activity columns, constraint-probability columns, and
objective-evaluation margin or satisfaction columns when available. It does
not include arbitrary metadata or the trajectory; read those from
`result.metadata` and `result.trajectory`.
