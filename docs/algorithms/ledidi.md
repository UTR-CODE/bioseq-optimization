# ledidi_optimize

Maximize the selected objective with an edit penalty, using seed-prior logits and hard Gumbel-softmax samples. Requires score_soft. Repeated proposals always consume budget; metadata.soft_query_count reports additional differentiable evaluations.

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
| `learning_rate` | 0.1 | Positive optimization step size |
| `ledidi_edit_penalty` | 0.1 | Nonnegative edit penalty per substitution |
| `ledidi_temperature` | 1.0 | Positive Gumbel-softmax temperature |
| `ledidi_epsilon` | 0.0001 | Positive seed-prior smoothing |

```python
from sequence_optimization import ledidi_optimize

result = ledidi_optimize(predictor, length=50, budget=200, num_candidates=4,
    objective=["maximize"], seed=7)
```

For gradient algorithms, use a [differentiable predictor](../custom_models.md#gradient-optimization).
The model itself is not trained. Each run returns unique sequences, objective scores,
raw predictions, query counts and a trajectory. Budgets exclude additional soft
forward/backward evaluations. [Objective parameters](../objectives.md).
