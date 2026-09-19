# genetic_algorithm_optimize

Elite selection, crossover and mutation. With `is_CDS_optimized=True`, crossover
and mutation operate on whole codons. Each initial CDS is optimized independently;
every result starts with `ATG`, ends with `TAA`, `TAG`, or `TGA`, and encodes the
same protein as that initial sequence. Only the standard genetic code is used.

For N parents, first take up to `parent_pool_multiplier * N` highest-scoring
members of the current feasible population. Keep the highest-scoring anchor,
then greedily add the candidate maximizing its minimum Hamming distance to
selected parents. The default multiplier is 5; 2 narrows the quality pool and
1 selects score-ranked elites. Parent count, crossover, mutation, and survivor
selection remain unchanged. This greedy diversity heuristic does not guarantee
a global optimum or prevent later generations from converging.

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
| `mutation_rate` | 0.02 | Mutation probability per position (0 < value <= 1); per codon for constrained CDS |
| `elite_fraction` | 0.2 | Elite proportion (0 < value <= 1) |
| `parent_pool_multiplier` | 5 | Candidate-pool multiplier before diverse parent selection |
| `is_CDS_optimized` | False | Validate and independently optimize each initial sequence as a protein-preserving CDS |

```python
from sequence_optimization import genetic_algorithm_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
result = genetic_algorithm_optimize(
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
