# den_optimize

Train a sequence generator through a differentiable predictor. Requires score_soft; repeated discrete proposals always consume budget.

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
| `generator_learning_rate` | 0.003 | Positive proposal-generator Adam learning rate |
| `den_seed_pretrain_epochs` | 100 | Nonnegative generator seed-fitting epochs |
| `den_diversity_weight` | 0.1 | Nonnegative similarity penalty |

```python
from sequence_optimization import den_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
result = den_optimize(
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
