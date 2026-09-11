# DeepSTARR enhancer activity

Load name: `deepstarr`.

The model accepts a 249-nt candidate and returns two outputs in this order:
`developmental`, `housekeeping`. The examples below maximize housekeeping while
minimizing developmental. The direction list is a search objective, not an
absolute activity threshold; see the [search workflow](../search_workflow.md)
and [objective options](../objectives.md).

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | cpu | Inference device |
| `checkpoint` | None | Optional checkpoint override |
| `weights_dir` | None | Optional model-weight root |
| `batch_size` | 4096 | Inference batch size |
| `pad_to_length` | None | Optional N-padding target; the model itself requires 249 nt |
| `padding_side` | right | left or right |
| `task_names` | model output names | Optional output labels; output order stays developmental, housekeeping |

## Black-box optimization

All nine black-box optimizers use the model's `score` method through the public
optimizer API. This shared setup creates four reproducible 249-nt starting
sequences. Each call states its own budget, initial population, and objective.
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

model = load_model("deepstarr", device="cpu")
initial = random_seed_sequences(249, count=4, seed=7)

random_result = random_optimize(
    model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
ism_result = ism_optimize(
    model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
annealing_result = simulated_annealing_optimize(
    model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
genetic_result = genetic_algorithm_optimize(
    model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
adalead_result = adalead_optimize(
    model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
basesearch_result = basesearch_optimize(
    model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
beam_result = beam_search_optimize(
    model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
cross_entropy_result = cross_entropy_optimize(
    model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
cbas_result = cbas_optimize(
    model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)

# To raise both outputs, use an all-maximize direction list:
all_high_result = genetic_algorithm_optimize(
    model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["maximize", "maximize"], seed=7,
)
```

Algorithm details: [random](../algorithms/random.md),
[ISM](../algorithms/ism_greedy.md), [simulated annealing](../algorithms/simulated_annealing.md),
[genetic algorithm](../algorithms/genetic_algorithm.md), [AdaLead](../algorithms/adalead.md),
[BaseSearch](../algorithms/basesearch.md), [beam search](../algorithms/beam_search.md),
[cross-entropy](../algorithms/cross_entropy_method.md), and [CbAS](../algorithms/cbas.md).

## Gradient optimization

The bundled PyTorch model uses channels-first `[batch, 4, length]` tensors.
`DifferentiableSequencePredictor` supplies the differentiable `score_soft`
adapter while retaining `model.score` for the final discrete candidates.

```python
from sequence_optimization import (
    DifferentiableSequencePredictor,
    den_optimize,
    fast_seqprop_optimize,
    ledidi_optimize,
    vanilla_activation_maximization_optimize,
)

gradient_model = DifferentiableSequencePredictor(
    model.model,
    onehot=model.encoder,
    device="cpu",
    task_names=model.task_names,
    soft_input_transform=lambda values: values.transpose(1, 2),
)

vanilla_result = vanilla_activation_maximization_optimize(
    gradient_model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
fast_seqprop_result = fast_seqprop_optimize(
    gradient_model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
ledidi_result = ledidi_optimize(
    gradient_model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
den_result = den_optimize(
    gradient_model, length=249, budget=1000, num_candidates=4,
    initial=initial, objective=["minimize", "maximize"], seed=7,
)
```

See [gradient model adapters](../custom_models.md#gradient-optimization) and the
[vanilla](../algorithms/vanilla_activation_maximization.md),
[Fast SeqProp](../algorithms/fast_seqprop.md), [Ledidi](../algorithms/ledidi.md),
and [DEN](../algorithms/den.md) references.
