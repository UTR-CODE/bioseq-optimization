# Optimus 5-Prime

Load name: `optimus5prime`.

Optimus 5-Prime accepts 50-nt sequences and returns mean ribosome load (MRL).
The examples below maximize the single MRL output on CPU from the same four
starting sequences.

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | cpu | Inference device |
| `checkpoint` | None | Optional checkpoint override |
| `weights_dir` | None | Optional model-weight root |
| `batch_size` | 4096 | Inference batch size |
| `pad_to_length` | None | Optional N-padding target; must satisfy model input requirements |
| `padding_side` | right | left or right |
| `task_names` | model output names | Optional output labels, without changing output order |

## Shared setup

Load the model through `bioseq_models`, then use it with the black-box
optimizers. `initial` must contain four unique 50-nt sequences.

```python
from bioseq_models import load_model
from sequence_optimization import (
    adalead_optimize,
    basesearch_optimize,
    beam_search_optimize,
    cbas_optimize,
    cross_entropy_optimize,
    den_optimize,
    fast_seqprop_optimize,
    genetic_algorithm_optimize,
    ism_optimize,
    ledidi_optimize,
    random_seed_sequences,
    random_optimize,
    simulated_annealing_optimize,
    vanilla_activation_maximization_optimize,
)

model = load_model("optimus5prime", device="cpu")

initial = random_seed_sequences(length=50, count=4, seed=7)

```

## Black-box optimizers

Each call below is independent and uses the five required optimizer arguments.

### Random search

```python
random_result = random_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(random_result.to_frame())
```

### Genetic algorithm

```python
genetic_result = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(genetic_result.to_frame())
```

### ISM greedy search

```python
ism_result = ism_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(ism_result.to_frame())
```

### Simulated annealing

```python
annealing_result = simulated_annealing_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(annealing_result.to_frame())
```

### AdaLead

```python
adalead_result = adalead_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(adalead_result.to_frame())
```

### Beam search

```python
beam_result = beam_search_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(beam_result.to_frame())
```

### Cross-entropy method

```python
cross_entropy_result = cross_entropy_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(cross_entropy_result.to_frame())
```

### BaseSearch

```python
basesearch_result = basesearch_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(basesearch_result.to_frame())
```

### CbAS

```python
cbas_result = cbas_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(cbas_result.to_frame())
```

## Gradient optimizers

The four gradient optimizers below use the configured
`DifferentiableSequencePredictor`, not the string-only model object.

```python
from sequence_optimization import DifferentiableSequencePredictor

gradient_predictor = DifferentiableSequencePredictor(
    model.model,
    onehot=model.encoder,
    device="cpu",
    task_names=model.task_names,
    soft_input_transform=lambda values: values.transpose(1, 2),
)
```

`gradient_predictor` keeps the model's encoder for discrete scoring and uses a
channels-first `[batch, 4, length]` tensor for soft scoring. The transpose is
needed because the differentiable predictor's canonical soft input is
`[batch, length, 4]`, while Optimus expects channels first.

### Vanilla activation maximization

```python
vanilla_result = vanilla_activation_maximization_optimize(
    gradient_predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(vanilla_result.to_frame())
```

### Fast SeqProp

```python
fast_seqprop_result = fast_seqprop_optimize(
    gradient_predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(fast_seqprop_result.to_frame())
```

### DEN

```python
den_result = den_optimize(
    gradient_predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(den_result.to_frame())
```

### Ledidi

```python
ledidi_result = ledidi_optimize(
    gradient_predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(ledidi_result.to_frame())
```

The gradient methods perform additional soft forward and backward passes;
those passes are not represented by the discrete `query_count`. See the
[search workflow](../search_workflow.md) for seed handling, cache behavior,
budgets, and saturation stopping. See [objective parameters](../objectives.md)
for direction lists, fixed scales, and bounds.

## Stop after the search plateaus

Use saturation screening when the stopping point should be based on observed
progress instead of a fixed `budget`. The optimizer still has a safety ceiling
through `maximum_queries`.

```python
from sequence_optimization import SaturationConfig, random_seed_sequences

initial = random_seed_sequences(length=50, count=4, seed=7)
saturated = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9),
        minimum_queries=40,
        plateau_patience_queries=80,
        maximum_queries=400,
        material_improvement=1e-4,
    ),
)

q80 = saturated[0.8]
q90 = saturated[0.9]
print(q80.to_frame())
print(q90.to_frame())
print(saturated.stop_reason, saturated.total_query_count)
```

q80 and q90 are selected after the run stops. They are the earliest retained
sets reaching 80% and 90% of the improvement between the first complete
archive and the final observed archive; they are not prediction percentiles.
See [saturation parameters and stop reasons](../search_workflow.md#stop-at-saturation).

Detailed parameters for each algorithm are available in the
[algorithm reference](../algorithms.md).
