# UTR-CODE: 5′UTR, 3′UTR and CDS

Load name: `utr_code`.

The model returns `log1p_rpf_rpkm_x5`. Maximize this translation-output proxy at
fixed RNA count and cell context. It is not a stability prediction.

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `cpu` | Inference device |
| `checkpoint` | `None` | Optional checkpoint override |
| `weights_dir` | `None` | Optional model-weight root |
| `design_region` | required | `utr5`, `cds`, or `utr3`; this region is supplied to `score()` |
| `utr5`, `cds`, `utr3` | `None` | The two regions that are not being designed |
| `cell_context` | `HEK293T` | Bundled context; another context requires `transcriptome_profile` |
| `transcriptome_profile` | `None` | 13,238 finite, non-negative raw expression values; the adapter applies `log1p(profile × 5)` |
| `rna_count` | `6.3` | Fixed non-negative RNA count |
| `batch_size` | `256` | Inference batch size |

`utr5` is limited to 1,000 nt, `cds` to 4,500 nt, and `utr3` to 4,000 nt.
When designing one region, pass both other regions and do not pass the designed
region. Replace the fixed strings in the examples with the corresponding
transcript sequences.

## 5′UTR design

The nine black-box optimizers operate on the fixed-context scorer. The examples
use four explicit 50-nt seeds.

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
    random_seed_sequences,
    random_optimize,
    simulated_annealing_optimize,
)

fixed_cds = "ATG" + "GCT" * 18 + "TAA"  # replace with the transcript CDS
fixed_utr3 = "ACGT" * 10                 # replace with the transcript 3′UTR
model_5utr = load_model(
    "utr_code",
    design_region="utr5",
    cds=fixed_cds,
    utr3=fixed_utr3,
    cell_context="HEK293T",
    rna_count=6.3,
    batch_size=256,
    device="cpu",
)
initial_5utr = random_seed_sequences(length=50, count=4, seed=7)

result_random = random_optimize(
    model_5utr, length=50, initial=initial_5utr, objective=["maximize"],
    num_candidates=4,
)
result_genetic = genetic_algorithm_optimize(
    model_5utr, length=50, initial=initial_5utr, objective=["maximize"],
    num_candidates=4,
)
result_ism = ism_optimize(
    model_5utr, length=50, initial=initial_5utr, objective=["maximize"],
    num_candidates=4,
)
result_annealing = simulated_annealing_optimize(
    model_5utr, length=50, initial=initial_5utr, objective=["maximize"],
    num_candidates=4,
)
result_adalead = adalead_optimize(
    model_5utr, length=50, initial=initial_5utr, objective=["maximize"],
    num_candidates=4,
)
result_beam = beam_search_optimize(
    model_5utr, length=50, initial=initial_5utr, objective=["maximize"],
    num_candidates=4,
)
result_cross_entropy = cross_entropy_optimize(
    model_5utr, length=50, initial=initial_5utr, objective=["maximize"],
    num_candidates=4,
)
result_basesearch = basesearch_optimize(
    model_5utr, length=50, initial=initial_5utr, objective=["maximize"],
    num_candidates=4,
)
result_cbas = cbas_optimize(
    model_5utr, length=50, initial=initial_5utr, objective=["maximize"],
    num_candidates=4,
)
```

The loaded UTR-CODE scorer exposes a hard-sequence `score()` interface. The four gradient
optimizers (`vanilla_activation_maximization`, `fast_seqprop`, `den`, and
`ledidi`) require a `score_soft()` adapter that preserves UTR-CODE's regulatory
padding, CDS padding, transcriptome profile, and RNA-count inputs. See
[custom model interfaces](../custom_models.md) when implementing that adapter.

## 3′UTR design

This is the same nine-optimizer interface with an 80-nt candidate and a fixed
5′UTR/CDS context.

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
    random_seed_sequences,
    random_optimize,
    simulated_annealing_optimize,
)

fixed_utr5 = "ACGT" * 10                 # replace with the transcript 5′UTR
fixed_cds = "ATG" + "GCT" * 18 + "TAA"  # replace with the transcript CDS
model_3utr = load_model(
    "utr_code",
    design_region="utr3",
    utr5=fixed_utr5,
    cds=fixed_cds,
    cell_context="HEK293T",
    rna_count=6.3,
    batch_size=256,
    device="cpu",
)
initial_3utr = random_seed_sequences(length=80, count=4, seed=7)

result_random = random_optimize(
    model_3utr, length=80, initial=initial_3utr, objective=["maximize"],
    num_candidates=4,
)
result_genetic = genetic_algorithm_optimize(
    model_3utr, length=80, initial=initial_3utr, objective=["maximize"],
    num_candidates=4,
)
result_ism = ism_optimize(
    model_3utr, length=80, initial=initial_3utr, objective=["maximize"],
    num_candidates=4,
)
result_annealing = simulated_annealing_optimize(
    model_3utr, length=80, initial=initial_3utr, objective=["maximize"],
    num_candidates=4,
)
result_adalead = adalead_optimize(
    model_3utr, length=80, initial=initial_3utr, objective=["maximize"],
    num_candidates=4,
)
result_beam = beam_search_optimize(
    model_3utr, length=80, initial=initial_3utr, objective=["maximize"],
    num_candidates=4,
)
result_cross_entropy = cross_entropy_optimize(
    model_3utr, length=80, initial=initial_3utr, objective=["maximize"],
    num_candidates=4,
)
result_basesearch = basesearch_optimize(
    model_3utr, length=80, initial=initial_3utr, objective=["maximize"],
    num_candidates=4,
)
result_cbas = cbas_optimize(
    model_3utr, length=80, initial=initial_3utr, objective=["maximize"],
    num_candidates=4,
)
```

## CDS design with synonymous constraints

For a CDS design that preserves the encoded protein, set
`is_CDS_optimized=True`. The nine black-box optimizers operate within the
synonymous codon space.

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

fixed_utr5 = "ACGT" * 10                 # replace with the transcript 5′UTR
fixed_utr3 = "ACGT" * 10                 # replace with the transcript 3′UTR
model_cds = load_model(
    "utr_code",
    design_region="cds",
    utr5=fixed_utr5,
    utr3=fixed_utr3,
    cell_context="HEK293T",
    rna_count=6.3,
    batch_size=256,
    device="cpu",
)
initial_cds = [
    "ATG" + "GCT" * 18 + "TAA",
    "ATG" + "GCT" * 17 + "GCC" + "TAA",
    "ATG" + "GCT" * 16 + "GCC" * 2 + "TAA",
    "ATG" + "GCT" * 15 + "GCC" * 3 + "TAA",
]
results_random = random_optimize(
    model_cds, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_genetic = genetic_algorithm_optimize(
    model_cds,
    length=60,
    initial=initial_cds,
    objective=["maximize"],
    num_candidates=4,
    is_CDS_optimized=True,
)
results_ism = ism_optimize(
    model_cds, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_annealing = simulated_annealing_optimize(
    model_cds, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_adalead = adalead_optimize(
    model_cds, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_beam = beam_search_optimize(
    model_cds, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_cross_entropy = cross_entropy_optimize(
    model_cds, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_basesearch = basesearch_optimize(
    model_cds, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)
results_cbas = cbas_optimize(
    model_cds, length=60, initial=initial_cds, objective=["maximize"],
    num_candidates=4, is_CDS_optimized=True,
)

print(results_genetic[0].to_frame())
```

Each variable is a list with one result per initial CDS, in input order. Every
candidate in a result translates to the same amino-acid sequence as that
result's initial CDS. The four gradient optimizers require a differentiable
UTR-CODE adapter with `score_soft()` and accept the same synonymous-CDS mode.

For fixed reference centers, scales, or bounds in a multi-output extension, see
[objectives](../objectives.md). The [search workflow](../search_workflow.md)
describes seed selection, budgets, and result inspection. See
[joint RPF and mRNA-stability optimization](../tasks/joint_rpf_stability.md) for
a protein-preserving CDS task with UTR-CODE and CDS Stability.
