# random_optimize

Uniform random candidate sampling with a fixed proposal budget. Random search
does not use saturation stopping because further independent samples can still
change the retained optimum without producing a meaningful convergence
trajectory.

| Parameter | Default | Meaning |
|---|---|---|
| `model` | required | Parameterized predictor accepting sequence strings |
| `length` | required | Candidate sequence length |
| `initial` | required | Starting sequences, or None for random initialization |
| `objective` | required | Per-output numeric targets or maximize, minimize, and ignore directions; an objective object is also accepted |
| `num_candidates` | required | Number of unique retained results |
| `budget` | 100000 | Number of candidate proposals, including the initial candidates |
| `saturation` | None | Not supported; use `budget` |
| `constraint` | None | One constraint callable or a list of independently intersected constraint callables |
| `diversity` | 5 | Minimum pairwise Hamming distance among returned sequences; zero disables it |
| `fixed_region` | None | List of zero-based positions or Python-style start:stop slices held to one shared template |
| `alphabet` | "ACGT" | Candidate alphabet |
| `seed` | 0 | Random seed |
| `count_repeated` | False | Random search always counts every proposal, including repeats |
| `is_CDS_optimized` | False | Independently optimize each initial sequence as a protein-preserving CDS |


```python
from sequence_optimization import random_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
result = random_optimize(
    predictor,
    length=50,
    initial=initial,
    objective=["maximize"],  # or [0.6] to stop after reaching score 0.6
    num_candidates=4,
    budget=10_000,
)
```

The optimizer scores the supplied `num_candidates` initial sequences, then
draws uniform random sequences until it has counted `budget` proposals. Repeated
sequences still consume the proposal budget, while cached sequences do not
cause another model call. A numeric objective target can stop the run before
the full budget is used.

For gradient algorithms, use a [differentiable predictor](../custom_models.md#custom-input).
The model itself is not trained. Each run returns unique sequences, objective scores,
raw predictions, query counts and a trajectory. Budgets exclude additional soft
forward/backward evaluations. [Objective parameters](../objectives.md).
