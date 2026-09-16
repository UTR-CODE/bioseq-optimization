# Synonymous CDS optimization

Optimized region: **CDS**.

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize

initial = [
    "ATG" + "GCT" * 18 + "TAA",
    "ATG" + "GCT" * 17 + "GCC" + "TAA",
    "ATG" + "GCT" * 16 + "GCC" * 2 + "TAA",
    "ATG" + "GCT" * 15 + "GCC" * 3 + "TAA",
]
model = load_model("cds_stability", species="human", device="cpu")
results = genetic_algorithm_optimize(
    model,
    length=len(initial[0]),
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    is_CDS_optimized=True,
)
for result in results:
    print(result.to_frame())
```

Use each actual CDS to preserve its protein sequence. Every initial must begin
with `ATG`, end with `TAA`, `TAG`, or `TGA`, contain no internal stops, and have a
length divisible by three. Synonymous codons come from the standard genetic
code; alternative genetic codes are not supported.

Pass `fixed_region` with zero-based positions or `"start:stop"` slices to keep
selected CDS bases unchanged. The synonymous choices are restricted at those
positions while every candidate continues to encode its initial protein.

Every initial and proposed candidate starts with `ATG`, ends with one of the
three stop codons, and translates to its own initial protein. Crossover and
mutation act on codons, with `mutation_rate` applied per codon. Each initial CDS
is optimized independently and must provide enough synonymous variants for
`num_candidates`. One initial returns one result; multiple initials return a
list in input order, with `num_candidates` candidates in each result. `budget`
and saturation stopping apply separately to each initial.

Every optimizer accepts `is_CDS_optimized=True`. Each one applies its search
operations within the synonymous codon space and supports fixed-budget or
saturation stopping.

For `random_optimize`, each sampled candidate contains a random number `N` of
different synonymous codon substitutions relative to its reference CDS, where
`0 < N < (length / 3 - 1)`. The initial `ATG` start codon is never changed;
every candidate remains in frame and preserves the reference protein sequence.

## Select candidates at saturation

Pass the same CDS constraints when using saturation stopping:

```python
from sequence_optimization import SaturationConfig

saturated_results = genetic_algorithm_optimize(
    model,
    length=len(initial[0]),
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    is_CDS_optimized=True,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9),
        minimum_queries=40,
        plateau_patience_queries=80,
        maximum_queries=800,
    ),
)
for saturated in saturated_results:
    q80 = saturated[0.8]
    q90 = saturated[0.9]
    print(q90.to_frame())
    print(saturated.stop_reason)
```

q80 and q90 retain each initial protein sequence. Their fractions refer to
observed objective improvement, not the fraction of changed codons.
[Initial sequences, budgets and saturation](../search_workflow.md).
