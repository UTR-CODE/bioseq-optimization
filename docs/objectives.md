# Objectives

Pass one direction per output: `maximize`, `minimize`, or `ignore`.
At least one output must be active. Lists use the worst signed output as score.

`MultiTaskObjective` accepts:

| Parameter | Default | Meaning |
|---|---|---|
| directions | None | Per-output direction list |
| high / low | empty | Selected output indices or task names; alternative to directions; high="all" selects all |
| aggregation | "worst" | worst, mean, or smoothmin |
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

## Relative task or model weights

Multi-output models and `CombinedScorer` use the same weighting method because
both present an `[N, T]` activity matrix to `MultiTaskObjective`. First obtain
one fixed reference scale per output. To assign positive relative weights
`weights`, divide those scales by the weights:

```python
import numpy as np

from sequence_optimization import MultiTaskObjective

reference_scales = np.array([0.4, 0.1, 2.0])
weights = np.array([2.0, 1.0, 0.5])
objective = MultiTaskObjective(
    directions=("maximize", "maximize", "minimize"),
    centers=(1.2, 0.7, 5.0),
    scales=reference_scales / weights,
    aggregation="mean",
)
```

This objective gives the first output twice the reference-normalized influence
of the second and the third half that influence. Only weight ratios affect
candidate ranking. All weights must be finite and positive.

With `aggregation="mean"`, improvements in one output can compensate for
declines in another. With `aggregation="worst"`, the optimizer raises the
smallest weighted margin instead; this is useful when every selected output
must improve. Bounds define success independently of weighting: a candidate
satisfies the objective only when every bounded margin reaches zero or above.

`TargetObjective(target=0, mode="maximize", penalty=1.0, other_weights=None)`
selects one output; mode="specificity" subtracts weighted off-target outputs
with penalty divided by the total number of tasks. target accepts a task index
or name. other_weights accepts a per-output list or index/name mapping.
