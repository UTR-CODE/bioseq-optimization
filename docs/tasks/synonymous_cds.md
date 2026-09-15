# Synonymous CDS optimization

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize

reference = "ATG" + "GCT" * 18 + "TAA"
allowed_codons = {"A": ["GCT", "GCC"]}
initial = [
    reference,
    "ATG" + "GCT" * 17 + "GCC" + "TAA",
    "ATG" + "GCT" * 16 + "GCC" * 2 + "TAA",
    "ATG" + "GCT" * 15 + "GCC" * 3 + "TAA",
]
model = load_model("cds_stability", species="human", device="cpu")
result = genetic_algorithm_optimize(
    model,
    length=len(reference),
    initial=initial,
    reference_cds=reference,
    synonymous_codons=allowed_codons,
    objective=["maximize"],
    num_candidates=4,
    budget=4000,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
print(result.to_frame())
```

Use the actual CDS to preserve its protein sequence. The reference must begin
with `ATG`, end with `TAA`, `TAG`, or `TGA`, and contain no internal stops. The
start codon remains fixed; the terminal codon may vary among allowed synonymous
stop codons. Each supplied amino-acid key restricts codons to its list; omitted
amino acids use all synonymous codons from the
[standard genetic code](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi#SG1).
Alternative genetic codes are not supported.

Pass `fixed_region` with zero-based positions or `"start:stop"` slices to keep
selected CDS bases unchanged. The allowed synonymous codons are restricted at
those positions while every candidate continues to encode the reference
protein.

Every initial and proposed candidate starts with `ATG`, ends with one of the
three stop codons, and translates to the reference protein. Crossover and
mutation act on codons, with `mutation_rate` applied per codon. Explicit initial
sequences must already satisfy the constraint; the reference need not be a
candidate if its internal codons are excluded by the custom table. The number
of allowed unique sequences must be at least num_candidates.

These constraints are supported by `genetic_algorithm_optimize`, including its
`saturation` parameter. Other algorithms reject constrained CDS requests.
Without `reference_cds`, sequence optimization does not preserve reading frames
or amino-acid sequences.

## Select candidates at saturation

Pass the same CDS constraints when using saturation stopping:

```python
from sequence_optimization import SaturationConfig

saturated = genetic_algorithm_optimize(
    model,
    length=len(reference),
    initial=initial,
    reference_cds=reference,
    synonymous_codons=allowed_codons,
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
