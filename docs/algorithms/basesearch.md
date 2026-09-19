# basesearch_optimize

BaseSearch now allocates 70% of proposal attempts by cluster score and recent
positive offspring-improvement ranks, and 30% by round-robin exploration.
Recent gains decay by 0.8 per batch. Each cluster retains at most 10 parents;
representatives within a selected cluster rotate. A cluster receives at most
20% of a nominal proposal batch (or 1/cluster_count when fewer than five
clusters exist, rounded up). Duplicate retries can start another quota cycle.
This quota constrains attempts, not accepted offspring or final output counts.

Parent scheduling now groups score-ordered candidates into deterministic
complete-link clusters with maximum Hamming diameter 20% of sequence length
(10 bases for 50-nt Optimus). Each cluster retains at most 10 expansion
representatives. The exploration portion rotates its starting cluster across
rounds. BaseSearch also prunes its queue by this cap. This limits search representatives, not the
number of final candidates per cluster. Several small output clusters are
acceptable; audit the largest and top-two cluster shares for concentration.

Priority-decayed parent selection alternates broad and single-base mutation.
Mutated candidates are scored in batches. Duplicates are not rescored. Stop on
budget exhaustion, exhausted space or duplicate stall. Queue priority does not
change returned objective scores.

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
| `is_CDS_optimized` | False | Independently optimize each initial sequence as a protein-preserving CDS |
| `mutation_rate` | 0.02 | Mutation probability per position (0 < value <= 1); per codon for constrained CDS |
| `basesearch_decay` | 0.995 | Parent priority decay, strictly between 0 and 1 |
| `basesearch_queue_size` | 1024 | Positive queue pruning target; prune when size exceeds twice this |
| `basesearch_max_stalled` | 1024 | Stop after this many consecutive duplicate proposals |
| `basesearch_batch_size` | 256 | Maximum number of mutated candidates scored together |

```python
from sequence_optimization import basesearch_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
result = basesearch_optimize(
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
