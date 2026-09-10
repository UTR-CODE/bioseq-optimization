# Biosequence Optimization

Design DNA/RNA sequences with a predictor and an optimization objective.

## Installation

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-optimization.git
```

For pretrained models used in the task guides:

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-models.git
```

## Algorithms

| Algorithm | Function | Predictor interface |
|---|---|---|
| [random](docs/algorithms/random.md) | `random_optimize` | `score` |
| [genetic_algorithm](docs/algorithms/genetic_algorithm.md) | `genetic_algorithm_optimize` | `score` |
| [ism_greedy](docs/algorithms/ism_greedy.md) | `ism_optimize` | `score` |
| [simulated_annealing](docs/algorithms/simulated_annealing.md) | `simulated_annealing_optimize` | `score` |
| [adalead](docs/algorithms/adalead.md) | `adalead_optimize` | `score` |
| [beam_search](docs/algorithms/beam_search.md) | `beam_search_optimize` | `score` |
| [cross_entropy_method](docs/algorithms/cross_entropy_method.md) | `cross_entropy_optimize` | `score` |
| [basesearch](docs/algorithms/basesearch.md) | `basesearch_optimize` | `score` |
| [cbas](docs/algorithms/cbas.md) | `cbas_optimize` | `score` |
| [vanilla_activation_maximization](docs/algorithms/vanilla_activation_maximization.md) | `vanilla_activation_maximization_optimize` | `score_soft` |
| [fast_seqprop](docs/algorithms/fast_seqprop.md) | `fast_seqprop_optimize` | `score_soft` |
| [den](docs/algorithms/den.md) | `den_optimize` | `score_soft` |
| [ledidi](docs/algorithms/ledidi.md) | `ledidi_optimize` | `score_soft` |

[Algorithm and model parameters](docs/algorithms.md).

## 1. Single task: custom CNN

The example weights measure nucleotide composition; use trained weights for a biological task.

```python
import torch
from torch import nn
from sequence_optimization import SequencePredictor, genetic_algorithm_optimize


class SingleOutputCNN(nn.Module):
    def __init__(self, base="G"):
        super().__init__()
        self.conv = nn.Conv1d(4, 1, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, "ACGT".index(base), 0] = 1
            self.conv.bias.fill_(-0.25)

    def forward(self, values):
        return self.conv(values).mean(dim=2)


predictor = SequencePredictor(SingleOutputCNN(), device="cpu")
result = genetic_algorithm_optimize(
    predictor, length=20, objective=["maximize"],
    num_candidates=4, budget=80, seed=7,
)
print(result.to_frame())
```

[Optimus: compare optimization algorithms](docs/models/optimus.md).

## 2. Multi-task: custom three-output CNN

```python
class ThreeOutputCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 3, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 2, 0] = 1
            self.conv.weight[1, 1, 0] = 1
            self.conv.weight[2, 3, 0] = 1
            self.conv.bias.fill_(-0.25)

    def forward(self, values):
        return self.conv(values).mean(dim=2)


predictor = SequencePredictor(ThreeOutputCNN(), device="cpu")
result = genetic_algorithm_optimize(
    predictor, length=20,
    objective=["maximize", "minimize", "ignore"],
    num_candidates=4, budget=80, seed=7,
)
print(result.to_frame())
```

Directions follow output order. `maximize` raises an output, `minimize` lowers it,
and `ignore` excludes it from the objective. Use three `maximize` entries for all-high.

[DeepSTARR: housekeeping specificity](docs/models/deepstarr.md) ·
[Malinois: cell-line specificity](docs/models/malinois.md).

## 3. Multiple independent models: custom CNNs

```python
from sequence_optimization import CombinedScorer

predictor = CombinedScorer(
    [SequencePredictor(SingleOutputCNN(base)) for base in "GCT"],
    names=["model_a", "model_b", "model_c"],
)
result = genetic_algorithm_optimize(
    predictor, length=20,
    objective=["maximize", "minimize", "minimize"],
    num_candidates=4, budget=80, seed=7,
)
print(result.to_frame())
```

Each model evaluates the same sequence. The objective list follows model order.

[Agarwal: three cell lines](docs/models/agarwal.md) ·
[DeepSTARR Mouse](docs/models/deepstarr_mouse.md) ·
[DeepSTARR-Embryo](docs/models/deepstarr_embryo.md).

[Custom CNN example functions](sequence_optimization/examples/custom_cnn_tasks.py).

## Model tasks

- [Optimus 5-Prime](docs/models/optimus.md)
- [UTR-CODE: 5′UTR, 3′UTR and CDS](docs/models/utr_code.md)
- [CDS stability](docs/models/cds_stability.md)
- [DeepSTARR: housekeeping high, developmental low](docs/models/deepstarr.md)
- [DREAM LegNet promoter activity](docs/models/dream_legnet.md)
- [Cetnar mRNA stability](docs/models/cetnar_stability.md)
- [Agarwal lentiMPRA cell specificity](docs/models/agarwal.md)
- [DeepSTARR Mouse tissue activity](docs/models/deepstarr_mouse.md)
- [DeepSTARR-Embryo tissue activity](docs/models/deepstarr_embryo.md)
- [Malinois cell-line-specific enhancers](docs/models/malinois.md)
- [Constrain CDS optimization to synonymous codons](docs/tasks/synonymous_cds.md)

## Inputs and results

`initial=[...]` supplies unique starting sequences; omit it to start randomly.
`num_candidates` controls seed and result count. `budget` limits candidate
evaluations/proposals; cached repeats can consume budget without another model
call. For combined models, each uncached candidate is evaluated by every model.
Additional differentiable evaluations are outside this budget.

`result.sequences` contains the designed strings, `result.scores` the objective,
and `result.activities` all raw predictions, including ignored outputs.

The default multi-output objective is the minimum signed prediction across
active outputs. For thresholds or fixed output normalization, see
[objective parameters](docs/objectives.md). Search directions alone do not
guarantee absolute activity levels.

## Use your own model

[Wrap your own model](docs/custom_models.md) ·
[Model input requirements](docs/model_input_requirements.md).
