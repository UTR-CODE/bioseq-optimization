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

`TargetObjective(target=0, mode="maximize", penalty=1.0, other_weights=None)`
selects one output; mode="specificity" subtracts weighted off-target outputs
with penalty divided by the total number of tasks. target accepts a task index
or name. other_weights accepts a per-output list or index/name mapping.
