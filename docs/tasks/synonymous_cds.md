# Synonymous CDS optimization

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize

reference = "ATG" + "GCT" * 18 + "TAA"
initial = [
    reference,
    "ATG" + "GCC" + "GCT" * 17 + "TAA",
    "ATG" + "GCC" * 2 + "GCT" * 16 + "TAA",
    "ATG" + "GCC" * 3 + "GCT" * 15 + "TAA",
]
model = load_model("cds_stability", species="human", device="cpu")
result = genetic_algorithm_optimize(
    model,
    length=len(reference),
    initial=initial,
    reference_cds=reference,
    synonymous_codons={"A": ["GCT", "GCC"]},
    objective=["maximize"],
    num_candidates=4,
    budget=200,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
print(result.to_frame())
```

Use the actual CDS to preserve its protein sequence. The reference must begin
with ATG, end with a stop codon and contain no internal stops. The first and
last codons remain fixed. Each supplied amino-acid key restricts internal
codons to its list; omitted amino acids use all synonymous codons from the
[standard genetic code](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi#SG1).
Alternative genetic codes are not supported.

Every initial and proposed candidate respects the codon table. Crossover and
mutation act on codons, with mutation_rate applied per codon. Explicit initial
sequences must already satisfy the constraint; the reference need not be a
candidate if its internal codons are excluded by the custom table. The number
of allowed unique sequences must be at least num_candidates.

These constraints are supported by genetic_algorithm, including its
optimize_to_saturation entry point. Other algorithms reject constrained CDS
requests. Without reference_cds, sequence optimization does not preserve reading
frames or amino-acid sequences.

## Select candidates at saturation

Pass the same CDS constraints when using saturation stopping:

```python
from sequence_optimization import SaturationConfig, optimize_to_saturation

saturated = optimize_to_saturation(
    model,
    length=len(reference),
    algorithm="genetic_algorithm",
    initial=initial,
    reference_cds=reference,
    synonymous_codons={"A": ["GCT", "GCC"]},
    objective=["maximize"],
    num_candidates=4,
    mutation_rate=0.1,
    elite_fraction=0.5,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9),
        minimum_queries=40,
        plateau_patience_queries=80,
        maximum_queries=800,
    ),
    seed=7,
)
q80 = saturated[0.8]
q90 = saturated[0.9]
print(q90.to_frame())
print(saturated.stop_reason)
```

q80 and q90 retain the same protein sequence and obey the same allowed-codon
table. Their fractions refer to observed objective improvement, not the fraction
of changed codons. [Initial sequences, budgets and saturation](../search_workflow.md).
