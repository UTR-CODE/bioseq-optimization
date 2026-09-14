# genetic_algorithm_optimize

Elite selection, crossover and mutation. With `reference_cds`, crossover and mutation operate on whole codons; every candidate starts with `ATG`, ends with `TAA`, `TAG`, or `TGA`, and encodes the reference protein. Only the standard genetic code is supported. Unspecified amino acids use all standard synonymous codons. A custom table restricts permitted codons, not amino-acid identity.

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
| `mutation_rate` | 0.02 | Mutation probability per position (0 < value <= 1); per codon for constrained CDS |
| `elite_fraction` | 0.2 | Elite proportion (0 < value <= 1) |
| `reference_cds` | None | Complete CDS beginning with ATG and ending with TAA, TAG, or TGA; length must match |
| `synonymous_codons` | None | Optional amino-acid to allowed codon lists; requires reference_cds |

```python
from sequence_optimization import genetic_algorithm_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
result = genetic_algorithm_optimize(
    predictor, length=50, initial=initial, budget=1000,
    num_candidates=4, mutation_rate=0.1, objective=["maximize"], seed=7,
)
```

For gradient algorithms, use a [differentiable predictor](../custom_models.md#gradient-optimization).
The model itself is not trained. Each run returns unique sequences, objective scores,
raw predictions, query counts and a trajectory. Budgets exclude additional soft
forward/backward evaluations. [Objective parameters](../objectives.md).
