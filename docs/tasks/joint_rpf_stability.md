# Joint RPF and mRNA-stability optimization

This task designs a coding sequence that raises both UTR-CODE's
`log1p_rpf_rpkm_x5` translation-output proxy and the CDS Stability model's
`stability_score`. Both models score the same complete CDS, while the search
changes codons without changing the encoded protein.

Only `genetic_algorithm_optimize` supports the synonymous-CDS search space.
The reference and every initial sequence must begin with `ATG`, end with
`TAA`, `TAG`, or `TGA`, contain no internal stop codon, and encode the same
amino-acid sequence.

## Models, transcript context, and initial CDSs

The example uses the Gluc CDS. Replace the fixed UTRs with its transcript
context. `reference_cds` restricts every candidate to the Gluc protein, and
`synonymous_codons` keeps `TAA` as the terminal stop codon. Amino acids omitted
from the mapping retain all synonymous codons in the standard genetic code.

```python
from bioseq_models import load_model
from sequence_optimization import CombinedScorer

fixed_utr5 = "ACGT" * 10  # replace with the transcript 5′UTR
fixed_utr3 = "ACGT" * 10  # replace with the transcript 3′UTR

# >Gluc_CDS
reference_cds = "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCTGTGGCCGAGGCCAAGCCCACCGAGAACAACGAAGACTTCAACATCGTGGCCGTGGCCAGCAACTTCGCGACCACGGATCTCGATGCTGACCGCGGGAAGTTGCCCGGCAAGAAGCTGCCGCTGGAGGTGCTCAAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGCTGTCTGATCTGCCTGTCCCACATCAAGTGCACGCCCAAGATGAAGAAGTTCATCCCAGGACGCTGCCACACCTACGAAGGCGACAAAGAGTCCGCACAGGGCGGCATAGGCGAGGCGATCGTCGACATTCCTGAGATTCCTGGGTTCAAGGACTTGGAGCCCATGGAGCAGTTCATCGCACAGGTCGATCTGTGTGTGGACTGCACAACTGGCTGCCTCAAAGGGCTTGCCAACGTGCAGTGTTCTGACCTGCTCAAGAAGTGGCTGCCGCAACGCTGTGCGACCTTTGCCAGCAAGATCCAGGGCCAGGTGGACAAGATCAAGGGGGCCGGTGGTGACTAA"
initial_cds = [
    reference_cds,
    "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGTATCGCTGTGGCCGAGGCCAAGCCCACCGAGAACAACGAAGACTTCAACATCGTGGCCGTAGCCAGCAACTTCGCGACCACGGATCTCGATGCTGACCGCGGGAAGTTGCCCGGCAAGAAACTGCCGCTGGAGGTGCTCAAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGAGGCTGTCTGATCTGCCTGTCCCACATCAAGTGCACGCCCAAGATGAAGAAGTTCATCCCCGGACGCTGCCACACCTACGAAGGCGACAAAGAGTCCGCACAGGGCGGCATAGGCGAGGCAATCGTCGACATTCCTGAGATTCCTGGGTTCAAGGACTTGGAGCCCATGGAGCAGTTCATAGCACAGGTCGATCTGTGTGTGGACTGCACAACTGGCTGCCTCAAAGGGCTTGCCAACGTACAGTGTTCTGACCTGCTCAAGAAGTGGCTGCCGCAACGCTGTGCGACCTTTGCCAGCAAAATCCAGGGCCAGGTGGACAAGATCAAGGGGGCCGGTGGTGACTAA",
    "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCTGTGGCCGAAGCCAAGCCCACCGAGAACAACGAAGACTTCAACATCGTGGCCGTGGCCAGCAACTTCGCAACCACGGATCTCGATGCTGACCGCGGGAAGTTGCCCGGCAAGAAGCTGCCGCTGGAGGTACTCAAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGCTGTCTGATCTGTCTGTCCCACATCAAGTGCACGCCCAAGATGAAGAAGTTCATCCCAGGACGCTGCCACACATACGAAGGCGACAAAGAGTCCGCACAGGGCGGCATAGGCGAGGCGATCGTCGACATTCCAGAGATTCCTGGGTTCAAGGACTTGGAGCCCATGGAGCAGTTCATCGCACAGGTCGATCTATGTGTGGACTGCACAACTGGCTGCCTCAAAGGGCTTGCCAACGTGCAGTGTTCTGACCTACTCAAGAAGTGGCTGCCGCAACGCTGTGCGACCTTTGCCAGCAAGATCCAGGGCCAGGTAGACAAGATCAAGGGGGCCGGTGGTGACTAA",
    "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCTGTGGCCGAGGCCAAGCCCACCGAAAACAACGAAGACTTCAACATCGTGGCCGTGGCCAGCAACTTCGCGACCACGGATCTCGACGCTGACCGCGGGAAGTTGCCCGGCAAGAAGCTGCCGCTGGAGGTGCTCAAAGAGATGGAGGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGCTGTCTGATCTGCCTGTCCCACATCAAATGCACGCCCAAGATGAAGAAGTTCATCCCAGGACGCTGCCACACCTACGAAGGCGACAAGGAGTCCGCACAGGGCGGCATAGGCGAGGCGATCGTCGACATTCCTGAGATTCCTGGGTTTAAGGACTTGGAGCCCATGGAGCAGTTCATCGCACAGGTCGATCTGTGTGTGGACTGCACCACTGGCTGCCTCAAAGGGCTTGCCAACGTGCAGTGTTCTGACCTGCTCAAGAAGTGGCTACCGCAACGCTGTGCGACCTTTGCCAGCAAGATCCAGGGCCAGGTGGACAAGATCAAGGGAGCCGGTGGTGACTAA",
]
allowed_codons = {"*": ["TAA"]}

rpf_model = load_model(
    "utr_code",
    design_region="cds",
    utr5=fixed_utr5,
    utr3=fixed_utr3,
    cell_context="HEK293T",
    rna_count=6.3,
    batch_size=256,
    device="cpu",
)
stability_model = load_model(
    "cds_stability",
    species="human",
    codonbert_model_dir=None,
    batch_size=4,
    device="cpu",
)
joint_model = CombinedScorer(
    [rpf_model, stability_model],
    names=["rpf", "mrna_stability"],
)
```

`CombinedScorer` orders the outputs as `[rpf, mrna_stability]`. The same order
is used by `objective` and `model_weights` below. The two equal weights produce
the mean of the signed RPF and mRNA-stability predictions.

## Fixed-budget search

```python
from sequence_optimization import genetic_algorithm_optimize

result = genetic_algorithm_optimize(
    joint_model,
    length=len(reference_cds),
    initial=initial_cds,
    reference_cds=reference_cds,
    synonymous_codons=allowed_codons,
    objective=["maximize", "maximize"],
    model_weights=[1.0, 1.0],
    num_candidates=4,
    budget=4_000,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
print(result.to_frame())
```

`activity_rpf` and `activity_mrna_stability` contain the two raw model
predictions. The candidate `budget` counts joint sequence evaluations; each
uncached candidate is evaluated once by each child model. Change the two
positive values in `model_weights` to adjust their relative influence.

## Saturation screening

Use `saturation` without `budget` to stop after the joint weighted-mean
trajectory levels off:

```python
from sequence_optimization import SaturationConfig, genetic_algorithm_optimize

saturated = genetic_algorithm_optimize(
    joint_model,
    length=len(reference_cds),
    initial=initial_cds,
    reference_cds=reference_cds,
    synonymous_codons=allowed_codons,
    objective=["maximize", "maximize"],
    model_weights=[1.0, 1.0],
    num_candidates=4,
    mutation_rate=0.1,
    elite_fraction=0.5,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9),
        minimum_queries=40,
        plateau_patience_queries=80,
        maximum_queries=4_000,
        material_improvement=0.01,
    ),
    seed=7,
)
q80 = saturated[0.8]
q90 = saturated[0.9]
print(q80.to_frame())
print(q90.to_frame())
print(saturated.stop_reason, saturated.total_query_count)
```

All retained sequences, including q80 and q90, translate to the same protein
as `reference_cds` and use only the configured synonymous codons. The q80 and
q90 fractions refer to the observed improvement in the joint weighted-mean
objective, not either raw model output alone.

[Synonymous CDS requirements](synonymous_cds.md) ·
[Multi-output objectives](../objectives.md) ·
[Initial sequences, budgets, and saturation](../search_workflow.md)
