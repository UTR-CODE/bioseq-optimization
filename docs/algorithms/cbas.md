# cbas_optimize

Fit an autoregressive proposal generator using discrete predictor scores. Repeated proposals always consume budget.

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
| `generator_pretrain_epochs` | 20 | Nonnegative generator pretraining epochs |
| `generator_update_epochs` | 5 | Nonnegative generator update epochs |
| `cbas_quantile` | 0.8 | Conditioning quantile, strictly between 0 and 1 |
| `cbas_batch_size` | 256 | Positive generated batch size |

```python
from sequence_optimization import cbas_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
result = cbas_optimize(
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
