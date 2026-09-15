# basesearch_optimize

Priority-decayed parent selection alternates broad and single-base mutation. Duplicates are not rescored. Stop on budget exhaustion, exhausted space or duplicate stall. Queue priority does not change returned objective scores.

| Parameter | Default | Meaning |
|---|---|---|
| `model` | required | Parameterized predictor accepting sequence strings |
| `length` | required | Candidate sequence length |
| `initial` | required | Starting sequences, or None for random initialization |
| `objective` | required | Per-output numeric targets or maximize, minimize, and ignore directions; an objective object is also accepted |
| `num_candidates` | required | Number of unique retained results |
| `budget` | 10,000 | Optional fixed candidate budget; invalid when saturation is set |
| `saturation` | None | SaturationConfig for plateau-based stopping; omit budget when set |
| `constraint` | None | One constraint callable or a list of independently intersected constraint callables |
| `diversity` | 5 | Minimum pairwise Hamming distance among returned sequences; zero disables it |
| `fixed_region` | None | List of zero-based positions or Python-style start:stop slices held to one shared template |
| `alphabet` | "ACGT" | Candidate alphabet |
| `seed` | 0 | Random seed |
| `count_repeated` | False | Charge repeated proposals even when cached |
| `mutation_rate` | 0.02 | Mutation probability per position (0 < value <= 1); per codon for constrained CDS |
| `basesearch_decay` | 0.995 | Parent priority decay, strictly between 0 and 1 |
| `basesearch_queue_size` | 1024 | Positive queue pruning target; prune when size exceeds twice this |
| `basesearch_max_stalled` | 1024 | Stop after this many consecutive duplicate proposals |

```python
from sequence_optimization import basesearch_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
result = basesearch_optimize(
    predictor,
    length=50,
    initial=initial,
    objective=["maximize"],  # or [0.6] to stop after reaching score 0.6
    num_candidates=4,
    budget=1000,
    seed=7,
)
```

For gradient algorithms, use a [differentiable predictor](../custom_models.md#gradient-optimization).
The model itself is not trained. Each run returns unique sequences, objective scores,
raw predictions, query counts and a trajectory. Budgets exclude additional soft
forward/backward evaluations. [Objective parameters](../objectives.md).
