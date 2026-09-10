# beam_search_optimize

Explore substitution neighborhoods of a beam. Width is max(1, round(num_candidates × elite_fraction)).

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
| `elite_fraction` | 0.2 | Elite proportion (0 < value <= 1) |

```python
from sequence_optimization import beam_search_optimize

result = beam_search_optimize(predictor, length=50, budget=200, num_candidates=4,
    objective=["maximize"], seed=7)
```

For gradient algorithms, use a [differentiable predictor](../custom_models.md#gradient-optimization).
The model itself is not trained. Each run returns unique sequences, objective scores,
raw predictions, query counts and a trajectory. Budgets exclude additional soft
forward/backward evaluations. [Objective parameters](../objectives.md).
