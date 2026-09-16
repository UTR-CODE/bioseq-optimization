# Custom three-output CNN

This recipe optimizes one sequence against three outputs produced by the same
PyTorch model. The example model reports the G, C and T nucleotide fractions,
each centered by subtracting `0.25`. Its weights are fixed demonstration
weights, not trained biological weights; load the weights for your own model
before creating the predictor.

## Model and predictor

The default `SequencePredictor` conversion produces A/C/G/T
one-hot tensors with shape `[batch, 4, length]`, which matches this CNN.
`SequencePredictor` can be passed to both discrete and gradient
optimizers because it provides `score` and `score_soft`.

```python
import torch
from torch import nn

from sequence_optimization import (
    SequencePredictor,
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
    random_optimize,
    simulated_annealing_optimize,
    vanilla_activation_maximization_optimize,
    random_seed_sequences,
)


class ThreeOutputCNN(nn.Module):
    """Demonstration model: centered G, C and T fractions."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 3, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 2, 0] = 1.0  # G fraction
            self.conv.weight[1, 1, 0] = 1.0  # C fraction
            self.conv.weight[2, 3, 0] = 1.0  # T fraction
            self.conv.bias.fill_(-0.25)

    def forward(self, values):
        return self.conv(values).mean(dim=2)


predictor = SequencePredictor(
    ThreeOutputCNN(),
    task_names=["G_fraction", "C_fraction", "T_fraction"],
    device="cpu",
)
```

The output order is `G_fraction`, `C_fraction`, `T_fraction`. Therefore an
objective list is positional: its first entry applies to G, its second to C,
and its third to T.

## Starting sequences

Create the same four reproducible 20-base sequences for each comparison:

```python
initial = random_seed_sequences(length=20, count=4, seed=7)
```

`initial` must contain at least `num_candidates` unique sequences. Omitting it
lets the optimizer draw `num_candidates` random starts.

## Goals and objective directions

The main goal maximizes G fraction, minimizes C fraction, and ignores T
fraction with `objective=["maximize", "minimize", "ignore"]`.

A finite number in the same position sets a raw-output lower target. For
example, `objective=[0.50, "minimize", "ignore"]` preserves G activity at or
above 0.50 after that target is reached while continuing to minimize C.

An ignored output remains available in `result.activities`, but it does not
affect the scalar objective. Active outputs use an equal-weight signed mean by
default. For fixed centers, scales, bounds, or a different aggregation, use the
options described in [objective parameters](../objectives.md).

## Run each optimizer

Choose an optimizer below. Each run starts from `initial`; the first nine
optimizers use `score`, and the last four also use `score_soft`.

```python
random_result = random_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

genetic_result = genetic_algorithm_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

ism_result = ism_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

annealing_result = simulated_annealing_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

adalead_result = adalead_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

beam_result = beam_search_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

cross_entropy_result = cross_entropy_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

basesearch_result = basesearch_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

cbas_result = cbas_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

vanilla_result = vanilla_activation_maximization_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

fast_seqprop_result = fast_seqprop_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

den_result = den_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)

ledidi_result = ledidi_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "ignore"], num_candidates=4,
)
```

See the [algorithm reference](../algorithms.md) for the behavior and
algorithm-specific parameters of each optimizer. Gradient optimizers perform
additional soft forward/backward evaluations outside the candidate `budget`.

## All-high goal with the genetic algorithm

To raise all three outputs, give all three positions the `maximize` direction:

```python
all_high_result = genetic_algorithm_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "maximize", "maximize"], num_candidates=4,
)
```

## Inspect results

Every result retains the designed strings, scalar objective scores, and raw
model outputs. `to_frame()` exposes them as columns:

| Result field or column | Meaning |
| --- | --- |
| `result.sequences` / `sequence` | One unique designed sequence per row |
| `result.scores` / `score` | Scalarized objective used by the optimizer |
| `result.activities` / `activity_G_fraction`, etc. | Raw G, C and T model outputs, including ignored outputs |
| `result.uncertainties` / `uncertainty` | Predictor uncertainty, when supplied |
| `result.query_count` | Candidate sequence evaluations charged to this run |

```python
print(genetic_result.to_frame())
```

For the difference between candidate budgets, soft-query accounting, and
multi-output model calls, see the [search workflow](../search_workflow.md).
