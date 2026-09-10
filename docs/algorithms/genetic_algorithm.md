# genetic_algorithm_optimize

Elite selection, crossover and mutation. With reference_cds, crossover and mutation operate on whole codons; protein sequence and terminal codons are preserved. Only the standard genetic code is supported. Unspecified amino acids use all standard synonymous codons. A custom table restricts permitted codons, not amino-acid identity.

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
| `mutation_rate` | 0.02 | Mutation probability per position (0 < value <= 1); per codon for constrained CDS |
| `elite_fraction` | 0.2 | Elite proportion (0 < value <= 1) |
| `reference_cds` | None | Complete ATG-to-stop reference CDS; length must match |
| `synonymous_codons` | None | Optional amino-acid to allowed codon lists; requires reference_cds |

```python
from sequence_optimization import genetic_algorithm_optimize

result = genetic_algorithm_optimize(predictor, length=50, budget=200, num_candidates=4,
    mutation_rate=0.1, objective=["maximize"], seed=7)
```

For gradient algorithms, use a [differentiable predictor](../custom_models.md#gradient-optimization).
The model itself is not trained. Each run returns unique sequences, objective scores,
raw predictions, query counts and a trajectory. Budgets exclude additional soft
forward/backward evaluations. [Objective parameters](../objectives.md).
