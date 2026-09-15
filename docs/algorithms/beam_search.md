# beam_search_optimize

Explore substitution neighborhoods of a beam. Width is max(1, round(num_candidates × elite_fraction)).

| Parameter | Default | Meaning |
|---|---|---|
| `scorer` | required | Predictor accepting sequence strings |
| `length` | required | Candidate sequence length |
| `budget` | 10,000 | Optional fixed candidate budget; invalid when saturation is set |
| `saturation` | None | SaturationConfig for plateau-based stopping; omit budget when set |
| `constraint` | None | One constraint callable or a list of independently intersected constraint callables |
| `num_candidates` | 100 | Number of unique seeds and retained results |
| `diversity` | 5 | Minimum pairwise Hamming distance among returned sequences; zero disables it |
| `fixed_region` | None | List of zero-based positions or Python-style start:stop slices held to one shared template |
| `initial` | None | Exactly num_candidates unique starting sequences; otherwise random |
| `objective` | None | Per-output numeric targets or maximize, minimize, and ignore directions; an objective object is also accepted |
| `model_weights` | None | Positive relative weights for direction-list outputs; default equal weights |
| `alphabet` | "ACGT" | Candidate alphabet |
| `seed` | 0 | Random seed |
| `count_repeated` | False | Charge repeated proposals even when cached |
| `elite_fraction` | 0.2 | Elite proportion (0 < value <= 1) |

```python
from sequence_optimization import beam_search_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
result = beam_search_optimize(
    predictor, length=50, initial=initial, budget=1000,
    num_candidates=4, objective=["maximize"], seed=7,
)
```

For gradient algorithms, use a [differentiable predictor](../custom_models.md#gradient-optimization).
The model itself is not trained. Each run returns unique sequences, objective scores,
raw predictions, query counts and a trajectory. Budgets exclude additional soft
forward/backward evaluations. [Objective parameters](../objectives.md).
