# vanilla_activation_maximization_optimize

Differentiate through soft positional probabilities. Requires score_soft; repeated discrete proposals always consume budget.

| Parameter | Default | Meaning |
|---|---|---|
| `scorer` | required | Predictor accepting sequence strings |
| `length` | required | Candidate sequence length |
| `budget` | 10,000 | Optional fixed candidate budget; invalid when saturation is set |
| `saturation` | None | SaturationConfig for plateau-based stopping; omit budget when set |
| `constraint` | None | One constraint callable or a list of independently intersected constraint callables |
| `num_candidates` | 100 | Number of unique seeds and retained results |
| `initial` | None | Exactly num_candidates unique starting sequences; otherwise random |
| `objective` | None | Output directions or objective object; default maximizes output zero |
| `model_weights` | None | Positive relative weights for direction-list outputs; default equal weights |
| `alphabet` | "ACGT" | Candidate alphabet |
| `seed` | 0 | Random seed |
| `count_repeated` | False | Charge repeated proposals even when cached |
| `learning_rate` | 0.1 | Positive optimization step size |

```python
from sequence_optimization import random_seed_sequences, vanilla_activation_maximization_optimize

initial = random_seed_sequences(length=50, count=4, seed=7)
result = vanilla_activation_maximization_optimize(
    predictor, length=50, initial=initial, budget=1000,
    num_candidates=4, objective=["maximize"], seed=7,
)
```

For gradient algorithms, use a [differentiable predictor](../custom_models.md#gradient-optimization).
The model itself is not trained. Each run returns unique sequences, objective scores,
raw predictions, query counts and a trajectory. Budgets exclude additional soft
forward/backward evaluations. [Objective parameters](../objectives.md).
