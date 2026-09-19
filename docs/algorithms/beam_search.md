# beam_search_optimize

Parent scheduling now groups score-ordered candidates into deterministic
complete-link clusters with maximum Hamming diameter 20% of sequence length
(10 bases for 50-nt Optimus). Each cluster retains at most 10 expansion
representatives. Selection visits clusters round-robin, taking one representative
per cluster before taking a second, with a rotating start across rounds.
AdaLead and beam apply this within their high-quality parent pool; BaseSearch
also prunes its queue by this cap. This limits search representatives, not the
number of final candidates per cluster. Several small output clusters are
acceptable; audit the largest and top-two cluster shares for concentration.

Explore substitution neighborhoods of a beam. Width is max(1, round(num_candidates × elite_fraction)).

For a beam of N parents, first collect up to `parent_pool_multiplier * N`
high-scoring candidates subject to the same `diversity` distance as returned
candidates. Keep the highest-scoring anchor, then select N parents by greedy
maximum minimum Hamming distance. Default multiplier is 5; use 2 for a narrower
pool or 1 for the previous score-ranked selection. The beam never exceeds N;
if too few feasible parents exist it uses the available smaller set. Zero
`diversity` removes the hard distance floor but retains diversity prioritization.

| Parameter | Default | Meaning |
|---|---|---|
| `model` | required | Parameterized predictor accepting sequence strings |
| `length` | required | Candidate sequence length |
| `initial` | required | Starting sequences, or None for random initialization |
| `objective` | required | Per-output maximize, minimize, or ignore directions; optional stop_values provides stopping lines; an objective object is also accepted |
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
| `elite_fraction` | 0.2 | Elite proportion (0 < value <= 1) |

```python
from sequence_optimization import beam_search_optimize, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
result = beam_search_optimize(
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
