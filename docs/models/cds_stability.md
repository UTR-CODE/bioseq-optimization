# CDS stability

Load name: `cds_stability`.

Optimized region: **CDS**.

The model returns `stability_score`; maximize it. Candidates must be complete
CDSs with an `ATG` start codon, a `TAA`, `TAG`, or `TGA` terminal codon, no
internal stops, and a length divisible by three. The adapter accepts at most
9,000 nt. The internal CodonBERT branch represents only the first 3,066 nt;
longer sequences are still accepted by the full model but are truncated in
that branch.

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `cpu` | Inference device |
| `checkpoint` | `None` | Optional stability checkpoint override |
| `weights_dir` | `None` | Optional model-weight root |
| `species` | `human` | `human`, `mouse`, `zebrafish`, `fruitfly`, or `yeast` |
| `codonbert_model_dir` | `None` | Local CodonBERT config/weights directory |
| `batch_size` | `4` | Inference batch size |

## CDS stability alone

For a protein-preserving CDS design, set `is_CDS_optimized=True`. The nine
black-box optimizers operate within the synonymous codon space. The four
gradient optimizers require a differentiable CDS Stability adapter with
`score_soft()`.

```python
from bioseq_models import load_model
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

model = load_model(
    "cds_stability",
    species="human",
    codonbert_model_dir=None,  # set a local directory when using local CodonBERT files
    batch_size=4,
    device="cpu",
)
initial_cds = [
    "ATG" + "GCT" * 18 + "TAA",
    "ATG" + "GCT" * 17 + "GCC" + "TAA",
    "ATG" + "GCT" * 16 + "GCC" * 2 + "TAA",
    "ATG" + "GCT" * 15 + "GCC" * 3 + "TAA",
]

results_random = random_optimize(
    model, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_genetic = genetic_algorithm_optimize(
    model,
    length=60,
    initial=initial_cds,
    objective=["maximize"],
    num_candidates=4,
    is_CDS_optimized=True,
)
results_ism = ism_optimize(
    model, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_annealing = simulated_annealing_optimize(
    model, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_adalead = adalead_optimize(
    model, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_beam = beam_search_optimize(
    model, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_cross_entropy = cross_entropy_optimize(
    model, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_basesearch = basesearch_optimize(
    model, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_cbas = cbas_optimize(
    model, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)

print(results_genetic[0].to_frame())
```

Each variable is a list with one result per initial CDS, in input order. Every
candidate in a result translates to the same amino-acid sequence as that
result's initial CDS.

## Joint stability and translation

[Joint RPF and mRNA-stability optimization](../tasks/joint_rpf_stability.md)
combines CDS Stability with UTR-CODE, calibrates the two output scales, and
preserves the reference protein through synonymous-codon search.
