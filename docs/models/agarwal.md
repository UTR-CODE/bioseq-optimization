# Agarwal lentiMPRA cell specificity

Load names: `agarwal_k562`, `agarwal_hepg2`, and `agarwal_wtc11`.

Each model accepts a 230-nt candidate and returns one `log2_rna_dna` output.
The combined scorer below evaluates the same candidate with the models in this
order: `K562`, `HepG2`, `WTC11`. Bundled loading uses ten cross-validation folds
and averages forward and reverse-complement predictions for each cell type.

The examples maximize K562 while minimizing HepG2 and WTC11. Directions are
search objectives, not absolute activity thresholds; see the
[search workflow](../search_workflow.md) and [objective options](../objectives.md).

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | cpu | Inference device |
| `checkpoint` | None | Optional checkpoint override; selects one checkpoint instead of the bundled ensemble |
| `weights_dir` | None | Optional model-weight root |
| `batch_size` | 4096 | Inference batch size |
| `pad_to_length` | None | Optional N-padding target; each model requires 230 nt |
| `padding_side` | right | left or right |
| `task_names` | `log2_rna_dna` | Optional name for each independent model output |

## Black-box optimization

All nine black-box optimizers use `CombinedScorer.score`, which calls each
loaded model's `score` method. Each call uses the five required optimizer
arguments.

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

k562 = load_model("agarwal_k562", device="cpu")
hepg2 = load_model("agarwal_hepg2", device="cpu")
wtc11 = load_model("agarwal_wtc11", device="cpu")
model = CombinedScorer([k562, hepg2, wtc11], names=["K562", "HepG2", "WTC11"])
initial = random_seed_sequences(230, count=4, seed=7)

random_result = random_optimize(
    model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
ism_result = ism_optimize(
    model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
annealing_result = simulated_annealing_optimize(
    model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
genetic_result = genetic_algorithm_optimize(
    model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
adalead_result = adalead_optimize(
    model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
basesearch_result = basesearch_optimize(
    model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
beam_result = beam_search_optimize(
    model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
cross_entropy_result = cross_entropy_optimize(
    model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
cbas_result = cbas_optimize(
    model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

# To raise K562, HepG2, and WTC11 together:
all_high_result = genetic_algorithm_optimize(
    model, length=230, initial=initial,
    objective=["maximize", "maximize", "maximize"], num_candidates=4,
)
```

Algorithm details: [random](../algorithms/random.md),
[ISM](../algorithms/ism_greedy.md), [simulated annealing](../algorithms/simulated_annealing.md),
[genetic algorithm](../algorithms/genetic_algorithm.md), [AdaLead](../algorithms/adalead.md),
[BaseSearch](../algorithms/basesearch.md), [beam search](../algorithms/beam_search.md),
[cross-entropy](../algorithms/cross_entropy_method.md), and [CbAS](../algorithms/cbas.md).

## Gradient optimization

The Agarwal modules use channels-first tensors in `A/G/C/T` order. Subclass
`SequencePredictor` to apply that channel order to both scoring paths. Each
ensemble's `.model` is the differentiable ten-fold, forward/reverse averaging
module.

```python
from sequence_optimization import (
    SequencePredictor,
    den_optimize,
    fast_seqprop_optimize,
    ledidi_optimize,
    vanilla_activation_maximization_optimize,
)

class AgarwalPredictor(SequencePredictor):
    def prepare_input(self, values):
        return values[..., [0, 2, 1, 3]].transpose(1, 2)

k562_gradient = AgarwalPredictor(
    k562.model,
    device="cpu",
    task_names=k562.task_names,
)
hepg2_gradient = AgarwalPredictor(
    hepg2.model,
    device="cpu",
    task_names=hepg2.task_names,
)
wtc11_gradient = AgarwalPredictor(
    wtc11.model,
    device="cpu",
    task_names=wtc11.task_names,
)
gradient_model = CombinedScorer(
    [k562_gradient, hepg2_gradient, wtc11_gradient],
    names=["K562", "HepG2", "WTC11"],
)

vanilla_result = vanilla_activation_maximization_optimize(
    gradient_model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
fast_seqprop_result = fast_seqprop_optimize(
    gradient_model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
ledidi_result = ledidi_optimize(
    gradient_model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
den_result = den_optimize(
    gradient_model, length=230, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
```

See [gradient model adapters](../custom_models.md#custom-input) and the
[vanilla](../algorithms/vanilla_activation_maximization.md),
[Fast SeqProp](../algorithms/fast_seqprop.md), [Ledidi](../algorithms/ledidi.md),
and [DEN](../algorithms/den.md) references.
