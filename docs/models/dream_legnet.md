# DREAM LegNet promoter activity

Load name: `dream_legnet`.

The model returns `promoter_activity`; maximize it. With
`variable_region_only=True`, candidates are 93-nt variable regions and the
adapter supplies the fixed context. With `False`, candidates are complete
110-nt constructs and the adapter still extracts the 93-nt variable region for
the published model input.

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `cpu` | Inference device |
| `checkpoint` | `None` | Optional checkpoint override |
| `weights_dir` | `None` | Optional model-weight root |
| `variable_region_only` | `False` | `True` for 93-nt variable regions; `False` for 110-nt constructs |
| `batch_size` | `4096` | Inference batch size |
| `pad_to_length` | `None` | Optional N-padding target; must meet model input requirements |
| `padding_side` | `right` | `left` or `right` |
| `task_names` | model output names | Optional labels; output order is unchanged |

## 93-nt variable-region design

The nine black-box optimizers use the adapter's fixed-context scorer. Every call
below uses the same four explicit seeds.

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
    "dream_legnet",
    variable_region_only=True,
    batch_size=4096,
    device="cpu",
)
initial = random_seed_sequences(length=93, count=4, seed=7)

result_random = random_optimize(
    model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_genetic = genetic_algorithm_optimize(
    model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_ism = ism_optimize(
    model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_annealing = simulated_annealing_optimize(
    model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_adalead = adalead_optimize(
    model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_beam = beam_search_optimize(
    model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_cross_entropy = cross_entropy_optimize(
    model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_basesearch = basesearch_optimize(
    model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_cbas = cbas_optimize(
    model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
```

To design complete 110-nt constructs, load `variable_region_only=False`, use
`length=110`, and pass four unique 110-nt initial constructs to the same nine
optimizer calls. The adapter accepts those complete constructs and extracts
the variable region before scoring.

## Gradient optimizers with the published context adapter

The LegNet adapter exposes a differentiable PyTorch model. The soft-input
transform below reproduces its published preprocessing: prepend the 57-nt
`CONTEXT_PREFIX`, reorder A/C/G/T to A/G/C/T, transpose to channels-first, and
append the singleton channel expected by LegNet. Keep `model.model`, which
contains the adapter's forward and reverse-complement average, rather than
calling the underlying network directly.

```python
import torch
from bioseq_models import load_model
from bioseq_models.adapters.dream_legnet import CONTEXT_PREFIX
from sequence_optimization import (
    SequencePredictor,
    den_optimize,
    fast_seqprop_optimize,
    ledidi_optimize,
    onehot,
    random_seed_sequences,
    vanilla_activation_maximization_optimize,
)

model = load_model(
    "dream_legnet",
    variable_region_only=True,
    batch_size=4096,
    device="cpu",
)
context = torch.from_numpy(onehot([CONTEXT_PREFIX], channels_first=False))

class DreamPredictor(SequencePredictor):
    def prepare_input(self, values):
        context_batch = context.to(values).expand(values.shape[0], -1, -1)
        dna = torch.cat([context_batch, values], dim=1)
        dna = dna[:, :, [0, 2, 1, 3]].transpose(1, 2)
        singleton = values.new_ones((values.shape[0], 1, 150))
        return torch.cat([dna, singleton], dim=1)

gradient_model = DreamPredictor(
    model.model,
    device="cpu",
    batch_size=4096,
    task_names=model.task_names,
)
initial = random_seed_sequences(length=93, count=4, seed=7)

result_vanilla = vanilla_activation_maximization_optimize(
    gradient_model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_fast_seqprop = fast_seqprop_optimize(
    gradient_model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_den = den_optimize(
    gradient_model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
result_ledidi = ledidi_optimize(
    gradient_model, length=93, initial=initial, objective=["maximize"],
    num_candidates=4,
)
```

The gradient adapter above is for the 93-nt variable-region workflow. A full
110-nt gradient adapter must first remove the adapter's 17-nt left flank with
`values[:, 17:, :]` before concatenating `CONTEXT_PREFIX`; the 93-nt transform
shown above should not be reused unchanged for 110-nt inputs.

The [objectives](../objectives.md) page documents directions, normalization,
and bounds. The [search workflow](../search_workflow.md) covers seed selection,
budgets, and result inspection.
