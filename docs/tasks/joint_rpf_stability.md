# Joint RPF and mRNA-stability optimization

Optimized region: **CDS**.

This task designs a coding sequence that raises both UTR-CODE's
`log1p_rpf_rpkm_x5` translation-output proxy and the CDS Stability model's
`stability_score`. Both models score the same complete CDS, while the search
changes codons without changing the encoded protein.

Every optimizer accepts the synonymous-CDS search space. Every initial sequence
must begin with `ATG`, end with `TAA`, `TAG`, or `TGA`, have a length divisible
by three, and contain no internal stop codon. Each is optimized in its own
protein-preserving search.

## Models, transcript context, and initial CDSs

The example uses the Gluc CDS. Replace the fixed UTRs with its transcript
context. Each initial CDS defines its protein-preserving synonymous search space
using the standard genetic code.

```python
from bioseq_models import load_model
from sequence_optimization import CombinedScorer

fixed_utr5 = "ACGT" * 10  # replace with the transcript 5′UTR
fixed_utr3 = "ACGT" * 10  # replace with the transcript 3′UTR

# >Gluc_CDS
cds = "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCTGTGGCCGAGGCCAAGCCCACCGAGAACAACGAAGACTTCAACATCGTGGCCGTGGCCAGCAACTTCGCGACCACGGATCTCGATGCTGACCGCGGGAAGTTGCCCGGCAAGAAGCTGCCGCTGGAGGTGCTCAAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGCTGTCTGATCTGCCTGTCCCACATCAAGTGCACGCCCAAGATGAAGAAGTTCATCCCAGGACGCTGCCACACCTACGAAGGCGACAAAGAGTCCGCACAGGGCGGCATAGGCGAGGCGATCGTCGACATTCCTGAGATTCCTGGGTTCAAGGACTTGGAGCCCATGGAGCAGTTCATCGCACAGGTCGATCTGTGTGTGGACTGCACAACTGGCTGCCTCAAAGGGCTTGCCAACGTGCAGTGTTCTGACCTGCTCAAGAAGTGGCTGCCGCAACGCTGTGCGACCTTTGCCAGCAAGATCCAGGGCCAGGTGGACAAGATCAAGGGGGCCGGTGGTGACTAA"
initial_cds = [
    cds,
    "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGTATCGCTGTGGCCGAGGCCAAGCCCACCGAGAACAACGAAGACTTCAACATCGTGGCCGTAGCCAGCAACTTCGCGACCACGGATCTCGATGCTGACCGCGGGAAGTTGCCCGGCAAGAAACTGCCGCTGGAGGTGCTCAAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGAGGCTGTCTGATCTGCCTGTCCCACATCAAGTGCACGCCCAAGATGAAGAAGTTCATCCCCGGACGCTGCCACACCTACGAAGGCGACAAAGAGTCCGCACAGGGCGGCATAGGCGAGGCAATCGTCGACATTCCTGAGATTCCTGGGTTCAAGGACTTGGAGCCCATGGAGCAGTTCATAGCACAGGTCGATCTGTGTGTGGACTGCACAACTGGCTGCCTCAAAGGGCTTGCCAACGTACAGTGTTCTGACCTGCTCAAGAAGTGGCTGCCGCAACGCTGTGCGACCTTTGCCAGCAAAATCCAGGGCCAGGTGGACAAGATCAAGGGGGCCGGTGGTGACTAA",
    "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCTGTGGCCGAAGCCAAGCCCACCGAGAACAACGAAGACTTCAACATCGTGGCCGTGGCCAGCAACTTCGCAACCACGGATCTCGATGCTGACCGCGGGAAGTTGCCCGGCAAGAAGCTGCCGCTGGAGGTACTCAAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGCTGTCTGATCTGTCTGTCCCACATCAAGTGCACGCCCAAGATGAAGAAGTTCATCCCAGGACGCTGCCACACATACGAAGGCGACAAAGAGTCCGCACAGGGCGGCATAGGCGAGGCGATCGTCGACATTCCAGAGATTCCTGGGTTCAAGGACTTGGAGCCCATGGAGCAGTTCATCGCACAGGTCGATCTATGTGTGGACTGCACAACTGGCTGCCTCAAAGGGCTTGCCAACGTGCAGTGTTCTGACCTACTCAAGAAGTGGCTGCCGCAACGCTGTGCGACCTTTGCCAGCAAGATCCAGGGCCAGGTAGACAAGATCAAGGGGGCCGGTGGTGACTAA",
    "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCTGTGGCCGAGGCCAAGCCCACCGAAAACAACGAAGACTTCAACATCGTGGCCGTGGCCAGCAACTTCGCGACCACGGATCTCGACGCTGACCGCGGGAAGTTGCCCGGCAAGAAGCTGCCGCTGGAGGTGCTCAAAGAGATGGAGGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGCTGTCTGATCTGCCTGTCCCACATCAAATGCACGCCCAAGATGAAGAAGTTCATCCCAGGACGCTGCCACACCTACGAAGGCGACAAGGAGTCCGCACAGGGCGGCATAGGCGAGGCGATCGTCGACATTCCTGAGATTCCTGGGTTTAAGGACTTGGAGCCCATGGAGCAGTTCATCGCACAGGTCGATCTGTGTGTGGACTGCACCACTGGCTGCCTCAAAGGGCTTGCCAACGTGCAGTGTTCTGACCTGCTCAAGAAGTGGCTACCGCAACGCTGTGCGACCTTTGCCAGCAAGATCCAGGGCCAGGTGGACAAGATCAAGGGAGCCGGTGGTGACTAA",
]

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

`CombinedScorer` orders the outputs as `[rpf, mrna_stability]`; this is also the
order used by `objective`. Active outputs use an equal-weight mean.

## Default search

```python
from sequence_optimization import (
    adalead_optimize,
    basesearch_optimize,
    beam_search_optimize,
    cbas_optimize,
    cross_entropy_optimize,
    genetic_algorithm_optimize,
    ism_optimize,
    random_optimize,
    simulated_annealing_optimize,
)

results_random = random_optimize(
    joint_model, length=len(initial_cds[0]), initial=initial_cds,
    objective=["maximize", "maximize"], num_candidates=4,
    is_CDS_optimized=True,
)
results_genetic = genetic_algorithm_optimize(
    joint_model,
    length=len(initial_cds[0]),
    initial=initial_cds,
    objective=["maximize", "maximize"],
    num_candidates=4,
    is_CDS_optimized=True,
)
results_ism = ism_optimize(
    joint_model, length=len(initial_cds[0]), initial=initial_cds,
    objective=["maximize", "maximize"], num_candidates=4,
    is_CDS_optimized=True,
)
results_annealing = simulated_annealing_optimize(
    joint_model, length=len(initial_cds[0]), initial=initial_cds,
    objective=["maximize", "maximize"], num_candidates=4,
    is_CDS_optimized=True,
)
results_adalead = adalead_optimize(
    joint_model, length=len(initial_cds[0]), initial=initial_cds,
    objective=["maximize", "maximize"], num_candidates=4,
    is_CDS_optimized=True,
)
results_beam = beam_search_optimize(
    joint_model, length=len(initial_cds[0]), initial=initial_cds,
    objective=["maximize", "maximize"], num_candidates=4,
    is_CDS_optimized=True,
)
results_cross_entropy = cross_entropy_optimize(
    joint_model, length=len(initial_cds[0]), initial=initial_cds,
    objective=["maximize", "maximize"], num_candidates=4,
    is_CDS_optimized=True,
)
results_basesearch = basesearch_optimize(
    joint_model, length=len(initial_cds[0]), initial=initial_cds,
    objective=["maximize", "maximize"], num_candidates=4,
    is_CDS_optimized=True,
)
results_cbas = cbas_optimize(
    joint_model, length=len(initial_cds[0]), initial=initial_cds,
    objective=["maximize", "maximize"], num_candidates=4,
    is_CDS_optimized=True,
)

print(results_genetic[0].to_frame())
```

`activity_rpf` and `activity_mrna_stability` contain the two raw model
predictions. Each uncached candidate is evaluated once by each child model.
Each result variable is a list in initial-CDS order. Plateau tracking is applied
separately to each initial CDS. The four gradient optimizers require
differentiable adapters for both child models.

## Saturation screening

Use `saturation` without `budget` to stop after the joint equal-weight mean
trajectory levels off:

```python
from sequence_optimization import SaturationConfig, genetic_algorithm_optimize

saturated_results = genetic_algorithm_optimize(
    joint_model,
    length=len(initial_cds[0]),
    initial=initial_cds,
    objective=["maximize", "maximize"],
    num_candidates=4,
    is_CDS_optimized=True,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9),
        minimum_queries=40,
        plateau_patience_queries=80,
        maximum_queries=4_000,
        material_improvement=0.01,
    ),
)
for saturated in saturated_results:
    q80 = saturated[0.8]
    q90 = saturated[0.9]
    print(q80.to_frame())
    print(q90.to_frame())
    print(saturated.stop_reason, saturated.total_query_count)
```

All retained sequences, including q80 and q90, translate to the same protein
as their corresponding initial CDS. The q80 and q90 fractions refer to the
observed improvement in the joint equal-weight objective, not either raw model
output alone.

[Synonymous CDS requirements](synonymous_cds.md) ·
[Multi-output objectives](../objectives.md) ·
[Initial sequences, budgets, and saturation](../search_workflow.md)
