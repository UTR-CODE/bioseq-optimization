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
reference_cds = "ATG" + "GCT" * 18 + "TAA"
allowed_codons = {"A": ["GCT", "GCC"]}
initial_cds = [
    reference_cds,
    "ATG" + "GCT" * 17 + "GCC" + "TAA",
    "ATG" + "GCT" * 16 + "GCC" * 2 + "TAA",
    "ATG" + "GCT" * 15 + "GCC" * 3 + "TAA",
]

result_constrained = genetic_algorithm_optimize(
    model,
    length=60,
    initial=initial_cds,
    reference_cds=reference_cds,
    synonymous_codons=allowed_codons,
    objective=["maximize"],
    num_candidates=4,
    budget=4000,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
```

## Joint stability and translation

This task scores each CDS with both CDS Stability and UTR-CODE. The fixed UTRs
below are placeholders; replace them with the actual transcript context. The
joint objective maximizes both `stability_score` and UTR-CODE's RPF output. A
protein-preserving joint search therefore uses the constrained genetic
optimizer shown here.

```python
from bioseq_models import load_model
from sequence_optimization import (
    CombinedScorer,
    genetic_algorithm_optimize,
)

fixed_utr5 = "ACGT" * 10  # replace with the transcript 5′UTR
fixed_utr3 = "ACGT" * 10  # replace with the transcript 3′UTR
reference_cds = "ATG" + "GCT" * 18 + "TAA"
allowed_codons = {"A": ["GCT", "GCC"]}
initial_cds = [
    reference_cds,
    "ATG" + "GCT" * 17 + "GCC" + "TAA",
    "ATG" + "GCT" * 16 + "GCC" * 2 + "TAA",
    "ATG" + "GCT" * 15 + "GCC" * 3 + "TAA",
]
stability = load_model(
    "cds_stability",
    species="human",
    codonbert_model_dir=None,
    batch_size=4,
    device="cpu",
)
translation = load_model(
    "utr_code",
    design_region="cds",
    utr5=fixed_utr5,
    utr3=fixed_utr3,
    cell_context="HEK293T",
    rna_count=6.3,
    batch_size=256,
    device="cpu",
)
joint_scorer = CombinedScorer(
    [stability, translation],
    names=["stability_score", "rpf"],
)
result_joint = genetic_algorithm_optimize(
    joint_scorer,
    length=60,
    initial=initial_cds,
    reference_cds=reference_cds,
    synonymous_codons=allowed_codons,
    objective=["maximize", "maximize"],
    num_candidates=4,
    budget=4000,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
```

The two outputs have different scales. Use fixed reference centers/scales or
bounds when the joint result needs a calibrated multi-output score; see
[objectives](../objectives.md). The [search workflow](../search_workflow.md)
covers fixed references, seeds, and result inspection.
