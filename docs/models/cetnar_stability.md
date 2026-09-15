# Cetnar mRNA stability

Load name: `cetnar_stability`.

The model returns `decay_rate_k_fit`. Minimize this decay rate for increased
predicted stability. The adapter accepts sequences up to 80 nt and right-pads
shorter sequences by default. This is a short-sequence decay surrogate, not a
complete-transcript stability model.

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `cpu` | Inference device |
| `checkpoint` | `None` | Optional checkpoint override |
| `weights_dir` | `None` | Optional model-weight root |
| `batch_size` | `4096` | Inference batch size |
| `pad_to_length` | `80` | N-padding target; the model input length is 80 |
| `padding_side` | `right` | `left` or `right` |
| `task_names` | model output names | Optional labels; output order is unchanged |

## Cetnar decay alone

Optimized region: **3′UTR**.

The nine black-box optimizers can minimize the decay output. Each call below
uses the same explicit four-sequence 80-nt seed set.

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

model = load_model(
    "cetnar_stability",
    pad_to_length=80,
    padding_side="right",
    batch_size=4096,
    device="cpu",
)
initial = random_seed_sequences(length=80, count=4, seed=7)

result_random = random_optimize(
    model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_genetic = genetic_algorithm_optimize(
    model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_ism = ism_optimize(
    model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_annealing = simulated_annealing_optimize(
    model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_adalead = adalead_optimize(
    model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_beam = beam_search_optimize(
    model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_cross_entropy = cross_entropy_optimize(
    model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_basesearch = basesearch_optimize(
    model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_cbas = cbas_optimize(
    model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
```

The bundled Cetnar PyTorch model uses channels-last input. A
`SequencePredictor` subclass applies that layout to discrete and soft scoring:

```python
from sequence_optimization import (
    SequencePredictor,
    den_optimize,
    fast_seqprop_optimize,
    ledidi_optimize,
    random_seed_sequences,
    vanilla_activation_maximization_optimize,
)

class ChannelsLastPredictor(SequencePredictor):
    def prepare_input(self, values):
        return values


gradient_model = ChannelsLastPredictor(
    model.model,
    device="cpu",
    batch_size=4096,
    task_names=model.task_names,
)
initial = random_seed_sequences(length=80, count=4, seed=7)
result_vanilla = vanilla_activation_maximization_optimize(
    gradient_model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_fast_seqprop = fast_seqprop_optimize(
    gradient_model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_den = den_optimize(
    gradient_model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
result_ledidi = ledidi_optimize(
    gradient_model, length=80, initial=initial, objective=["minimize"],
    num_candidates=4,
)
```

Use CPU for this gradient example. The soft input has canonical shape
`[batch, length, A/C/G/T]`; the adapter's `_RestoredDecay` transposes it for the
CNN-BiLSTM and restores the measured decay-rate scale.

## Joint decay and UTR-CODE translation

Optimized region: **3′UTR**.

The joint scorer places the same 80-nt candidate in UTR-CODE's 3′UTR slot with
fixed 5′UTR/CDS placeholders. Replace those strings with the transcript
context. The objective minimizes Cetnar decay and maximizes UTR-CODE RPF.

```python
from bioseq_models import load_model
from sequence_optimization import CombinedScorer

fixed_utr5 = "ACGT" * 10                 # replace with the transcript 5′UTR
fixed_cds = "ATG" + "GCT" * 18 + "TAA"  # replace with the transcript CDS
cetnar = load_model(
    "cetnar_stability",
    pad_to_length=80,
    padding_side="right",
    batch_size=4096,
    device="cpu",
)
translation = load_model(
    "utr_code",
    design_region="utr3",
    utr5=fixed_utr5,
    cds=fixed_cds,
    cell_context="HEK293T",
    rna_count=6.3,
    batch_size=256,
    device="cpu",
)
joint_scorer = CombinedScorer(
    [cetnar, translation],
    names=["decay_rate", "rpf"],
)
initial_joint = random_seed_sequences(length=80, count=4, seed=7)
```

Each black-box optimizer can use this two-output scorer:

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

result_random = random_optimize(
    joint_scorer, length=80, initial=initial_joint,
    objective=["minimize", "maximize"], num_candidates=4,
)
result_genetic = genetic_algorithm_optimize(
    joint_scorer, length=80, initial=initial_joint,
    objective=["minimize", "maximize"], num_candidates=4,
)
result_ism = ism_optimize(
    joint_scorer, length=80, initial=initial_joint,
    objective=["minimize", "maximize"], num_candidates=4,
)
result_annealing = simulated_annealing_optimize(
    joint_scorer, length=80, initial=initial_joint,
    objective=["minimize", "maximize"], num_candidates=4,
)
result_adalead = adalead_optimize(
    joint_scorer, length=80, initial=initial_joint,
    objective=["minimize", "maximize"], num_candidates=4,
)
result_beam = beam_search_optimize(
    joint_scorer, length=80, initial=initial_joint,
    objective=["minimize", "maximize"], num_candidates=4,
)
result_cross_entropy = cross_entropy_optimize(
    joint_scorer, length=80, initial=initial_joint,
    objective=["minimize", "maximize"], num_candidates=4,
)
result_basesearch = basesearch_optimize(
    joint_scorer, length=80, initial=initial_joint,
    objective=["minimize", "maximize"], num_candidates=4,
)
result_cbas = cbas_optimize(
    joint_scorer, length=80, initial=initial_joint,
    objective=["minimize", "maximize"], num_candidates=4,
)
```

The joint `CombinedScorer` uses the black-box optimizers because the loaded
UTR-CODE scorer does not expose `score_soft()`. Gradient optimization requires
a UTR-CODE soft-input adapter that preserves its complete transcript context.
For fixed output centers, scales, or bounds, see [objectives](../objectives.md).
The [search workflow](../search_workflow.md) covers multi-output result
inspection.
