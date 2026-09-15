# CDS stability

Load name: `cds_stability`.

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

For a protein-preserving CDS design, use synonymous codon constraints. Only
`genetic_algorithm_optimize` is compatible with this constraint; the other
eight black-box optimizers and the four gradient optimizers reject constrained
CDS spaces because their proposal representations do not preserve codons.

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize

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

results = genetic_algorithm_optimize(
    model,
    length=60,
    initial=initial_cds,
    objective=["maximize"],
    num_candidates=4,
    is_CDS_optimized=True,
    budget=4000,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
for result in results:
    print(result.to_frame())
```

## Joint stability and translation

[Joint RPF and mRNA-stability optimization](../tasks/joint_rpf_stability.md)
combines CDS Stability with UTR-CODE, calibrates the two output scales, and
preserves the reference protein through synonymous-codon search.
