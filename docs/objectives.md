# Objectives

Pass one entry per output. Each entry may be `maximize`, `minimize`, `ignore`,
or a finite numeric target. A numeric entry is a minimum in raw model-output
units. At least one output must be active.

Before a numeric target is reached, its margin contributes to the normalized
weighted objective. Once a candidate reaches it, later selection keeps that
output at or above the target and continues optimizing the other entries. With
only numeric targets and `ignore` entries, the search stops when one scored
candidate satisfies every numeric target. A fixed budget remains the safety
ceiling. Numeric targets cannot be combined with saturation stopping.

Direction lists use the normalized weighted mean of active signed outputs.
Pass `model_weights` directly to an optimizer; omitting it gives every active
output equal weight.

```python
from sequence_optimization import genetic_algorithm_optimize

targeted = genetic_algorithm_optimize(
    scorer,
    length=50,
    objective=[0.8, "maximize", "ignore"],
    num_candidates=100,
    budget=10_000,
)
```

`MultiTaskObjective` accepts:

| Parameter | Default | Meaning |
|---|---|---|
| directions | None | Per-output numeric target or maximize, minimize, and ignore entry |
| high / low | empty | Selected output indices or task names; alternative to directions; high="all" selects all |
| aggregation | "mean" | mean, worst, or smoothmin |
| centers | None | Fixed per-output reference centers; default zero |
| scales | None | Fixed positive per-output scales; default one |
| model_weights | None | Non-negative relative weights in output order; active weights must be positive |
| lower_bounds | None | Raw-output minimum for each high task |
| upper_bounds | None | Raw-output maximum for each low task |
| temperature | 0.1 | Positive smoothmin temperature |

Bounds must cover all selected outputs and replace centers. Margins are
`(prediction - lower_bound) / scale` for high tasks and
`(upper_bound - prediction) / scale` for low tasks. All bounds are satisfied
when `worst_margin >= 0`. Mean and smoothmin scores do not certify this.
Without bounds, directions guide search but do not define absolute success.
Compute normalization from a fixed reference dataset, not each candidate batch.
Numeric entries are a compact alternative for raw-output lower targets and
cannot be combined with `lower_bounds` or `upper_bounds` in the same
`MultiTaskObjective`.

## Relative task or model weights

Multi-output models and `CombinedScorer` use the same optimizer parameter
because both present an `[N, T]` activity matrix. The weight order matches the
model-output order:

```python
from sequence_optimization import genetic_algorithm_optimize

result = genetic_algorithm_optimize(
    scorer,
    length=50,
    objective=["maximize", "maximize", "minimize"],
    model_weights=[2.0, 1.0, 0.5],
    num_candidates=100,
    budget=10_000,
)
```

This objective gives the first output twice the influence of the second and the
third half that influence. Only weight ratios affect candidate ranking. Active
weights must be finite and positive. Supply one weight per model output; an
ignored output may use zero.

For outputs with different units or numeric ranges, weights can account for
both relative importance and scale in the simple interface. Use explicit
`scales` in `MultiTaskObjective` when normalization and biological importance
need to be configured separately. With `aggregation="mean"`, improvements in
one output can compensate for declines in another. With
`aggregation="worst"`, the optimizer raises the smallest weighted margin.
Bounds define success independently of weighting: a candidate satisfies the
objective only when every bounded margin reaches zero or above.

`TargetObjective(target=0, mode="maximize", penalty=1.0, other_weights=None)`
selects one output; mode="specificity" subtracts weighted off-target outputs
with penalty divided by the total number of tasks. target accepts a task index
or name. other_weights accepts a per-output list or index/name mapping.
