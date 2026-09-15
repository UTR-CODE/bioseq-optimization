# DeepSTARR-Embryo tissue activity

Load names: `deepstarr_embryo_<tissue>_activity`, where `<tissue>` is one of
`brain_vnc`, `muscle`, `epidermis`, or `gut`.

Optimized region: **enhancer**.

Each activity model accepts a 1001-nt candidate and returns one probability.
The combined scorer below evaluates models in this order: `brain_vnc`, `muscle`,
`epidermis`, `gut`. The examples maximize brain_vnc while minimizing the other
three tissues. Directions are search objectives, not absolute activity
thresholds; see the [search workflow](../search_workflow.md) and
[objective options](../objectives.md).

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | cpu | Inference device |
| `checkpoint` | None | Optional checkpoint override |
| `weights_dir` | None | Optional model-weight root |
| `batch_size` | 4096 | Inference batch size |
| `pad_to_length` | None | Optional N-padding target; each model requires 1001 nt |
| `padding_side` | right | left or right |
| `task_names` | model output name | Optional label; output order in the combined scorer is listed above |

The separate `brain_specific_activity` model predicts a specialized endpoint;
it is not an additional tissue in this four-output comparison. Accessibility
models exist for brain, cns, epidermis, midgut, somatic_muscle, and
visceral_muscle and are separate endpoints.

## Black-box optimization

All nine black-box optimizers use `CombinedScorer.score`, which calls each
model's `score` method. The shared setup creates four reproducible 1001-nt
sequences; each call uses the five required optimizer arguments.

```python
from bioseq_models import load_model
from sequence_optimization import (
    CombinedScorer,
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

brain_vnc = load_model("deepstarr_embryo_brain_vnc_activity", device="cpu")
muscle = load_model("deepstarr_embryo_muscle_activity", device="cpu")
epidermis = load_model("deepstarr_embryo_epidermis_activity", device="cpu")
gut = load_model("deepstarr_embryo_gut_activity", device="cpu")
model = CombinedScorer(
    [brain_vnc, muscle, epidermis, gut],
    names=["brain_vnc", "muscle", "epidermis", "gut"],
)
initial = random_seed_sequences(1001, count=4, seed=7)

random_result = random_optimize(
    model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
ism_result = ism_optimize(
    model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
annealing_result = simulated_annealing_optimize(
    model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
genetic_result = genetic_algorithm_optimize(
    model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
adalead_result = adalead_optimize(
    model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
basesearch_result = basesearch_optimize(
    model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
beam_result = beam_search_optimize(
    model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
cross_entropy_result = cross_entropy_optimize(
    model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
cbas_result = cbas_optimize(
    model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)

# To raise all four tissue activities together:
all_high_result = genetic_algorithm_optimize(
    model, length=1001, initial=initial,
    objective=["maximize", "maximize", "maximize", "maximize"], num_candidates=4,
)
```

Algorithm details: [random](../algorithms/random.md),
[ISM](../algorithms/ism_greedy.md), [simulated annealing](../algorithms/simulated_annealing.md),
[genetic algorithm](../algorithms/genetic_algorithm.md), [AdaLead](../algorithms/adalead.md),
[BaseSearch](../algorithms/basesearch.md), [beam search](../algorithms/beam_search.md),
[cross-entropy](../algorithms/cross_entropy_method.md), and [CbAS](../algorithms/cbas.md).

## Gradient optimization

The tissue adapters use channels-first `[batch, 4, 1001]` tensors. Wrap each
underlying PyTorch module in a `SequencePredictor`, then combine
the differentiable scorers in the same output order.

```python
from sequence_optimization import (
    SequencePredictor,
    den_optimize,
    fast_seqprop_optimize,
    ledidi_optimize,
    vanilla_activation_maximization_optimize,
)

brain_vnc_gradient = SequencePredictor(
    brain_vnc.model, device="cpu",
    task_names=brain_vnc.task_names,
)
muscle_gradient = SequencePredictor(
    muscle.model, device="cpu",
    task_names=muscle.task_names,
)
epidermis_gradient = SequencePredictor(
    epidermis.model, device="cpu",
    task_names=epidermis.task_names,
)
gut_gradient = SequencePredictor(
    gut.model, device="cpu",
    task_names=gut.task_names,
)
gradient_model = CombinedScorer(
    [brain_vnc_gradient, muscle_gradient, epidermis_gradient, gut_gradient],
    names=["brain_vnc", "muscle", "epidermis", "gut"],
)

vanilla_result = vanilla_activation_maximization_optimize(
    gradient_model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
fast_seqprop_result = fast_seqprop_optimize(
    gradient_model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
ledidi_result = ledidi_optimize(
    gradient_model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
den_result = den_optimize(
    gradient_model, length=1001, initial=initial,
    objective=["maximize", "minimize", "minimize", "minimize"], num_candidates=4,
)
```

See [gradient model adapters](../custom_models.md#custom-input) and the
[vanilla](../algorithms/vanilla_activation_maximization.md),
[Fast SeqProp](../algorithms/fast_seqprop.md), [Ledidi](../algorithms/ledidi.md),
and [DEN](../algorithms/den.md) references.
