# Objectives

Pass one entry per output: `maximize`, `minimize`, or `ignore` only.
At least one output must be active. Set numeric stopping lines separately with
`stop_values`; the default is a list of `None` (unlimited) for every output.

Before a stopping line is reached, its margin contributes to the objective.
For maximize, reach or exceed the line; for minimize, reach or fall below it.
Once reached, selection maintains that threshold while optimizing other tasks.
When every active task has a line, search stops when one scored candidate
satisfies all lines jointly. An ignored output must have a `None` line.
Predictions are not clipped: discrete steps may overshoot a line.
A positive `budget` can provide a
fixed query limit; otherwise the default plateau rules apply.

Direction lists use the equal-weight mean of active signed outputs.

```python
from sequence_optimization import genetic_algorithm_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=100, seed=7)
targeted = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize", "minimize", "ignore"],
    stop_values=[0.8, -2.0, None],
    num_candidates=100,
)
```

`MultiTaskObjective` accepts:

| Parameter | Default | Meaning |
|---|---|---|
| directions | None | Per-output maximize, minimize, or ignore entry |
| stop_values | all None | Directional stopping lines, in model-output order |
| high / low | empty | Selected output indices or task names; alternative to directions; high="all" selects all |
| aggregation | "mean" | mean, worst, or smoothmin |
| centers | None | Fixed per-output reference centers; default zero |
| scales | None | Fixed positive per-output scales; default one |
| lower_bounds | None | Raw-output minimum for each high task |
| upper_bounds | None | Raw-output maximum for each low task |
| temperature | 0.1 | Positive smoothmin temperature |

Bounds must cover all selected outputs and replace centers. Margins are
`(prediction - lower_bound) / scale` for high tasks and
`(upper_bound - prediction) / scale` for low tasks. All bounds are satisfied
when `worst_margin >= 0`. Mean and smoothmin scores do not certify this.
Without bounds, directions guide search but do not define absolute success.
Compute normalization from a fixed reference dataset, not each candidate batch.
Stopping lines cannot be combined with `lower_bounds` or `upper_bounds` in the same
`MultiTaskObjective`.

For outputs with different units or numeric ranges, use fixed positive `scales`
in `MultiTaskObjective`. With `aggregation="mean"`, improvements in one output
can compensate for declines in another. With `aggregation="worst"`, the
optimizer raises the smallest margin. Bounds define success independently of
aggregation: a candidate satisfies the objective only when every bounded margin
reaches zero or above.

`TargetObjective(target=0, mode="maximize", penalty=1.0, other_weights=None)`
selects one output; mode="specificity" subtracts weighted off-target outputs
with penalty divided by the total number of tasks. target accepts a task index
or name. other_weights accepts a per-output list or index/name mapping.
