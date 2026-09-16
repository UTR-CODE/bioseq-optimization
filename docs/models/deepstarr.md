# DeepSTARR enhancer activity

Load name: `deepstarr`.

Optimized region: **enhancer**.

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
sequences. Each call uses the five required optimizer arguments.

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
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
ism_result = ism_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
annealing_result = simulated_annealing_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
genetic_result = genetic_algorithm_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
adalead_result = adalead_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
basesearch_result = basesearch_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
beam_result = beam_search_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
cross_entropy_result = cross_entropy_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
cbas_result = cbas_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)

# To raise both outputs, use an all-maximize direction list:
all_high_result = genetic_algorithm_optimize(
    model, length=249, initial=initial,
    objective=["maximize", "maximize"], num_candidates=4,
)
```

Algorithm details: [random](../algorithms/random.md),
[ISM](../algorithms/ism_greedy.md), [simulated annealing](../algorithms/simulated_annealing.md),
[genetic algorithm](../algorithms/genetic_algorithm.md), [AdaLead](../algorithms/adalead.md),
[BaseSearch](../algorithms/basesearch.md), [beam search](../algorithms/beam_search.md),
[cross-entropy](../algorithms/cross_entropy_method.md), and [CbAS](../algorithms/cbas.md).

## Gradient optimization

The loaded model accepts canonical A/C/G/T soft probabilities for gradient
optimization.

```python
from sequence_optimization import (
    den_optimize,
    fast_seqprop_optimize,
    ledidi_optimize,
    vanilla_activation_maximization_optimize,
)

vanilla_result = vanilla_activation_maximization_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
fast_seqprop_result = fast_seqprop_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
ledidi_result = ledidi_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
den_result = den_optimize(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"], num_candidates=4,
)
```

See [gradient model adapters](../custom_models.md#custom-input) and the
[vanilla](../algorithms/vanilla_activation_maximization.md),
[Fast SeqProp](../algorithms/fast_seqprop.md), [Ledidi](../algorithms/ledidi.md),
and [DEN](../algorithms/den.md) references.
