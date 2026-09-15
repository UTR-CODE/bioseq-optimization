# Three independent custom CNNs

This recipe combines three independent single-output CNNs. Every child scores
the same candidate sequence, and the resulting columns are ordered by the
`CombinedScorer` list. The demonstration weights measure G, C and T fractions
centered by subtracting `0.25`; they are fixed toy weights, not trained
biological models. Replace them with your own trained weights for an
application.

## Model classes and combined predictor

```python
import torch
from torch import nn

from sequence_optimization import (
    CombinedScorer,
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


class SingleOutputCNN(nn.Module):
    """Demonstration model returning one centered nucleotide fraction."""

    def __init__(self, base):
        super().__init__()
        self.conv = nn.Conv1d(4, 1, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, "ACGT".index(base), 0] = 1.0
            self.conv.bias.fill_(-0.25)

    def forward(self, values):
        return self.conv(values).mean(dim=2)


predictor_g = SequencePredictor(
    SingleOutputCNN("G"), task_names=["G_fraction"], device="cpu"
)
predictor_c = SequencePredictor(
    SingleOutputCNN("C"), task_names=["C_fraction"], device="cpu"
)
predictor_t = SequencePredictor(
    SingleOutputCNN("T"), task_names=["T_fraction"], device="cpu"
)

predictor = CombinedScorer(
    [predictor_g, predictor_c, predictor_t],
    names=["G_fraction", "C_fraction", "T_fraction"],
)
```

`SequencePredictor` is used for every child, so the combined
scorer supports both the discrete `score` interface and the gradient
`score_soft` interface. A candidate is sent to the G, C and T models in that
order. The objective list uses the same order.

## Starting sequences

Create four reproducible 20-base sequences for each run:

```python
initial = random_seed_sequences(length=20, count=4, seed=7)
```

The starting sequences must be unique and their count must equal
`num_candidates`.

## Goals and normalization

The independent-model goal maximizes the G model and minimizes the C and T
models with `objective=["maximize", "minimize", "minimize"]`.

The objective is positional: the first direction applies to `predictor_g`,
the second to `predictor_c`, and the third to `predictor_t`. The combined
scorer returns all three raw outputs for every same-sequence evaluation. The
active signed outputs use an equal-weight mean. For fixed centers, scales,
bounds, or another aggregation, use the options in
[objective parameters](../objectives.md).

A finite number may replace a direction to set a raw-output lower target for
that model. After the target is reached, later selection keeps candidates at
or above it while continuing to optimize the remaining directions.

## Run each optimizer

Choose an optimizer below. Each run starts from `initial`; the first nine use
the black-box `score` method and the last four also use differentiable
`score_soft`.

```python
random_result = random_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

genetic_result = genetic_algorithm_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

ism_result = ism_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

annealing_result = simulated_annealing_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

adalead_result = adalead_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

beam_result = beam_search_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

cross_entropy_result = cross_entropy_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

basesearch_result = basesearch_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

cbas_result = cbas_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

vanilla_result = vanilla_activation_maximization_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

fast_seqprop_result = fast_seqprop_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

den_result = den_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)

ledidi_result = ledidi_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "minimize", "minimize"], num_candidates=4,
)
```

See the [algorithm reference](../algorithms.md) for algorithm behavior and
optional parameters. Gradient optimizers make additional soft model queries
outside the discrete candidate `budget`.

## All-high goal with the genetic algorithm

To maximize all three independent model outputs, change all objective entries
to `maximize`:

```python
all_high_result = genetic_algorithm_optimize(
    predictor, length=20, initial=initial,
    objective=["maximize", "maximize", "maximize"], num_candidates=4,
)
```

## Budget and result accounting

`CombinedScorer` applies each child model to the same sequence. The optimizer's
`budget` is the candidate/proposal budget for the run, while each child keeps a
cumulative sequence-query count. Thus one uncached candidate contributes one
query to each of the three children, not one shared model query. By contrast,
a single model with three outputs returns all three columns from one model
evaluation. See the [search workflow](../search_workflow.md) for the full
candidate, child-query, cache, and soft-query accounting rules.

Each result exposes:

| Result field or column | Meaning |
| --- | --- |
| `result.sequences` / `sequence` | Unique sequences shared by all three models |
| `result.scores` / `score` | Scalarized maximize/minimize objective |
| `result.activities` / `activity_G_fraction`, etc. | Raw output from each child model |
| `result.uncertainties` / `uncertainty` | Combined uncertainty, when all children provide it |
| `result.query_count` | Candidate evaluations charged to the optimizer run |

```python
print(genetic_result.to_frame())
```
