# cross_entropy_optimize

Fit positional categorical distributions to elites. Here learning_rate is the smoothing factor, clipped to at most 1.

| Parameter | Default | Meaning |
|---|---|---|
| `model` | required | Parameterized predictor accepting sequence strings |
| `length` | required | Candidate sequence length |
| `initial` | required | Starting sequences, or None for random initialization |
| `objective` | required | Per-output numeric targets or maximize, minimize, and ignore directions; an objective object is also accepted |
| `num_candidates` | required | Number of unique retained results |
| `budget` | None (unlimited) | Optional fixed candidate budget; omitted runs to a plateau and returns q99 |
| `saturation` | None | Optional SaturationConfig for custom plateau rules and gain fractions; omit budget when set |
| `constraint` | None | One constraint callable or a list of independently intersected constraint callables |
| `diversity` | 5 | Minimum pairwise Hamming distance among returned sequences; zero disables it |
| `fixed_region` | None | List of zero-based positions or Python-style start:stop slices held to one shared template |
| `alphabet` | "ACGT" | Candidate alphabet |
| `seed` | 0 | Random seed |
| `count_repeated` | False | Charge repeated proposals even when cached |
| `is_CDS_optimized` | False | Independently optimize each initial sequence as a protein-preserving CDS |
| `cross_entropy_batch_size` | 256 | Minimum number of categorical samples proposed per round |
| `elite_fraction` | 0.2 | Elite proportion (0 < value <= 1) |
| `learning_rate` | 0.1 | Positive optimization step size |

```python
from sequence_optimization import cross_entropy_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
result = cross_entropy_optimize(
    predictor,
    length=50,
    initial=initial,
    objective=["maximize"],  # or [0.6] to stop after reaching score 0.6
    num_candidates=4,
)
```

For gradient algorithms, use a [differentiable predictor](../custom_models.md#custom-input).
The model itself is not trained. Each run returns unique sequences, objective scores,
raw predictions, query counts and a trajectory. Budgets exclude additional soft
forward/backward evaluations. [Objective parameters](../objectives.md).
