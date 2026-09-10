# Malinois cell-line-specific enhancers

Load name: `malinois`.

The candidate is a 200-nt variable enhancer. The adapter inserts a fixed 200-nt
left context and a fixed 200-nt right context before prediction, so the neural
network receives 600 nt. Outputs are ordered `K562_log2FC`, `HepG2_log2FC`,
`SKNSH_log2FC`.

The examples below maximize K562 while minimizing HepG2 and SKNSH. Directions
are search objectives, not absolute activity thresholds; see the
[search workflow](../search_workflow.md) and [objective options](../objectives.md).

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | cpu | Inference device |
| `checkpoint` | None | Optional checkpoint override |
| `weights_dir` | None | Optional model-weight root |
| `batch_size` | 4096 | Inference batch size |
| `pad_to_length` | None | Optional N-padding target; candidate inputs must be 200 nt |
| `padding_side` | right | left or right |
| `task_names` | model output names | Optional output labels; output order stays K562, HepG2, SKNSH |

## Black-box optimization

All nine black-box optimizers use `model.score` and optimize only the 200-nt
candidate. Each call states its budget, initial population, and objective.
The 1000-query examples leave enough budget for the initial four sequences and
one complete DNA single-mutant neighborhood in ISM.

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

model = load_model("malinois", device="cpu")
initial = random_seed_sequences(200, count=4, seed=7)

random_result = random_optimize(
    model, length=200, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
ism_result = ism_optimize(
    model, length=200, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
annealing_result = simulated_annealing_optimize(
    model, length=200, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
genetic_result = genetic_algorithm_optimize(
    model, length=200, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
adalead_result = adalead_optimize(
    model, length=200, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
basesearch_result = basesearch_optimize(
    model, length=200, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
beam_result = beam_search_optimize(
    model, length=200, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
cross_entropy_result = cross_entropy_optimize(
    model, length=200, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
cbas_result = cbas_optimize(
    model, length=200, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)

# To raise K562, HepG2, and SKNSH together:
all_high_result = genetic_algorithm_optimize(
    model, length=200, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "maximize", "maximize"], seed=7,
)
```

Algorithm details: [random](../algorithms/random.md),
[ISM](../algorithms/ism_greedy.md), [simulated annealing](../algorithms/simulated_annealing.md),
[genetic algorithm](../algorithms/genetic_algorithm.md), [AdaLead](../algorithms/adalead.md),
[BaseSearch](../algorithms/basesearch.md), [beam search](../algorithms/beam_search.md),
[cross-entropy](../algorithms/cross_entropy_method.md), and [CbAS](../algorithms/cbas.md).

## Gradient optimization

Gradient methods need a differentiable adapter that preserves Malinois's fixed
contexts. The hard-path encoder remains `model.encoder`; the soft path below
concatenates the same contexts to canonical `[batch, 200, A/C/G/T]` tensors and
then converts them to the model's channels-first `[batch, 4, 600]` layout.

```python
import torch

from bioseq_models.adapters.malinois import LEFT_CONTEXT, RIGHT_CONTEXT
from sequence_optimization import (
    DifferentiableSequencePredictor,
    den_optimize,
    fast_seqprop_optimize,
    ledidi_optimize,
    onehot,
    vanilla_activation_maximization_optimize,
)

left = torch.from_numpy(onehot([LEFT_CONTEXT], channels_first=False))
right = torch.from_numpy(onehot([RIGHT_CONTEXT], channels_first=False))

def add_malinois_context(values):
    count = values.shape[0]
    left_batch = left.to(values).expand(count, -1, -1)
    right_batch = right.to(values).expand(count, -1, -1)
    return torch.cat((left_batch, values, right_batch), dim=1).transpose(1, 2)

gradient_model = DifferentiableSequencePredictor(
    model.model,
    onehot=model.encoder,
    device="cpu",
    task_names=model.task_names,
    soft_input_transform=add_malinois_context,
)

vanilla_result = vanilla_activation_maximization_optimize(
    gradient_model, length=200, budget=100, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
fast_seqprop_result = fast_seqprop_optimize(
    gradient_model, length=200, budget=100, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
ledidi_result = ledidi_optimize(
    gradient_model, length=200, budget=100, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
den_result = den_optimize(
    gradient_model, length=200, budget=100, num_candidates=4,
    initial=initial, objective=["maximize", "minimize", "minimize"], seed=7,
)
```

See [gradient model adapters](../custom_models.md#gradient-optimization) and the
[vanilla](../algorithms/vanilla_activation_maximization.md),
[Fast SeqProp](../algorithms/fast_seqprop.md), [Ledidi](../algorithms/ledidi.md),
and [DEN](../algorithms/den.md) references.
