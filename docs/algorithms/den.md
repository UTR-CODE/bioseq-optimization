# den_optimize

Train a sequence generator through a differentiable predictor. Requires score_soft; repeated discrete proposals always consume budget.

| Parameter | Default | Meaning |
|---|---|---|
| `scorer` | required | Predictor accepting sequence strings |
| `length` | required | Candidate sequence length |
| `budget` | required | Candidate evaluation/proposal budget, at least num_candidates |
| `num_candidates` | 100 | Number of unique seeds and retained results |
| `initial` | None | Exactly num_candidates unique starting sequences; otherwise random |
| `objective` | None | Output directions or objective object; default maximizes output zero |
| `alphabet` | "ACGT" | Candidate alphabet |
| `seed` | 0 | Random seed |
| `count_repeated` | False | Charge repeated proposals even when cached |
| `generator_learning_rate` | 0.003 | Positive proposal-generator Adam learning rate |
| `den_seed_pretrain_epochs` | 100 | Nonnegative generator seed-fitting epochs |
| `den_diversity_weight` | 0.1 | Nonnegative similarity penalty |

```python
from sequence_optimization import den_optimize

result = den_optimize(predictor, length=50, budget=200, num_candidates=4,
    objective=["maximize"], seed=7)
```

For gradient algorithms, use a [differentiable predictor](../custom_models.md#gradient-optimization).
The model itself is not trained. Each run returns unique sequences, objective scores,
raw predictions, query counts and a trajectory. Budgets exclude additional soft
forward/backward evaluations. [Objective parameters](../objectives.md).
