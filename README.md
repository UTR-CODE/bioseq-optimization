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


single_model = SequencePredictor(SingleOutputCNN(), device="cpu")
initial = ["ACGT" * 5, "AAGT" * 5, "ACCT" * 5, "AGGT" * 5]
single_result = genetic_algorithm_optimize(
    single_model,
    length=20,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    budget=200,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
print(single_result.to_frame())
```

[Optimus: compare optimization algorithms](docs/models/optimus.md).

## 2. Multi-task: custom three-output CNN

One model returns three columns. Here they measure G, C and T composition,
respectively. Raise the first, lower the second and leave the third out of the
optimization objective:

```python
import torch
from torch import nn
from sequence_optimization import SequencePredictor, genetic_algorithm_optimize


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


multi_model = SequencePredictor(
    ThreeOutputCNN(),
    task_names=["G_activity", "C_activity", "T_activity"],
    device="cpu",
)
initial = ["ACGT" * 5, "AAGT" * 5, "ACCT" * 5, "AGGT" * 5]
specific_result = genetic_algorithm_optimize(
    multi_model,
    length=20,
    initial=initial,
    objective=["maximize", "minimize", "ignore"],
    num_candidates=4,
    budget=200,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
print(specific_result.to_frame())
```

Directions follow output order. `maximize` raises an output, `minimize` lowers it,
and `ignore` excludes it from the objective. Ignored outputs are still reported
in `result.activities`; they are not constrained to remain unchanged.

To raise all three outputs, run a separate search from the same initial sequences:

```python
all_high_result = genetic_algorithm_optimize(
    multi_model,
    length=20,
    initial=initial,
    objective=["maximize", "maximize", "maximize"],
    num_candidates=4,
    budget=200,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
print(all_high_result.activities)
```

The default direction-list score is the smallest signed prediction: the
weakest active output determines the score. This encourages balanced progress,
but conflicting model predictions can prevent all goals from improving together.
For fixed output normalization or activity thresholds, see
[objective parameters](docs/objectives.md).

[Multi-task CNN with every optimizer](docs/tasks/multi_task.md) ·
[DeepSTARR: housekeeping specificity](docs/models/deepstarr.md) ·
[Malinois: cell-line specificity](docs/models/malinois.md).

## 3. Multiple independent models: custom CNNs

```python
from sequence_optimization import CombinedScorer, SequencePredictor, genetic_algorithm_optimize

model_a = SequencePredictor(SingleOutputCNN("G"), device="cpu")
model_b = SequencePredictor(SingleOutputCNN("C"), device="cpu")
model_c = SequencePredictor(SingleOutputCNN("T"), device="cpu")
combined = CombinedScorer(
    [model_a, model_b, model_c],
    names=["model_a", "model_b", "model_c"],
)
initial = ["ACGT" * 5, "AAGT" * 5, "ACCT" * 5, "AGGT" * 5]
combined_result = genetic_algorithm_optimize(
    combined,
    length=20,
    initial=initial,
    objective=["maximize", "minimize", "minimize"],
    num_candidates=4,
    budget=200,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
print(combined_result.to_frame())
print(combined.model_query_counts)
```

`SingleOutputCNN` is defined in the first example. Each model evaluates the same
sequence and returns one output; each can use its own input conversion. The
objective list follows the order in `CombinedScorer`, so this search raises
model A's prediction and lowers B's and C's.

`budget=200` applies to combined candidate evaluations, not 200 split between
the models. Each uncached candidate is sent to all three models. The counters
above report cumulative discrete sequence evaluations per model.

Use `objective=["maximize", "maximize", "maximize"]` for an all-high search,
or `ignore` to exclude a model's output from the objective. An ignored model is
still evaluated. Use fixed reference scales when outputs have different units.

[Independent CNNs with every optimizer](docs/tasks/multiple_models.md) ·
[Agarwal: three cell lines](docs/models/agarwal.md) ·
[DeepSTARR Mouse](docs/models/deepstarr_mouse.md) ·
[DeepSTARR-Embryo](docs/models/deepstarr_embryo.md).

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

## Initial sequences

`initial` supplies the starting sequences. Provide exactly `num_candidates`
unique strings, each matching `length`. Omit `initial` to sample random starting
sequences, or generate a shared starting set before comparing optimizers:

```python
from sequence_optimization import random_seed_sequences

initial = random_seed_sequences(length=20, count=4, seed=7)
```

`load_seed_sequences("seeds.txt", length=20, count=4)` reads one sequence per
line. Blank lines and lines starting with `#` are ignored.

## Budget and results

`num_candidates=4` sets the starting population and the number of unique
candidates retained in the result. `budget=200` limits counted candidate
evaluations, including the initial population; it is not the number of epochs
or the result size. A run may stop before exhausting the budget.

With the default counting mode, the nine discrete optimizers except CbAS count
uncached sequences. CbAS and the four gradient optimizers count repeated
proposals too. `count_repeated=True` selects proposal counting for other methods.
Cache hits can therefore consume budget without another model call.
Gradient evaluations are additional work outside this discrete budget.

`result.sequences` contains the designed strings, `result.scores` the objective,
and `result.activities` all raw predictions, including ignored outputs.

## Saturation screening

Using `single_model` and `initial` from the single-task example, call
`optimize_to_saturation` to stop after progress levels off. It tracks the
mean objective of the best `num_candidates` unique sequences found so far.

```python
from sequence_optimization import SaturationConfig, optimize_to_saturation

saturated = optimize_to_saturation(
    single_model,
    length=20,
    algorithm="genetic_algorithm",
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9),
        minimum_queries=40,
        plateau_patience_queries=80,
        maximum_queries=800,
        material_improvement=1e-4,
    ),
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
q80 = saturated[0.8]
q90 = saturated[0.9]
print(q80.to_frame())
print(q90.to_frame())
print(saturated.stop_reason, saturated.total_query_count)
```

q80 and q90 are the earliest retained candidate sets reaching 80% and 90% of
the improvement from the first complete archive to the final observed archive.
They are not score percentiles or guarantees of reaching the global optimum.
The run must finish before these fractions can be selected. `maximum_queries`
is a safety ceiling; an optimizer can also stop on its own before a plateau is
detected. Check `stop_reason` to distinguish these cases.

[Initial sequences, budget accounting and saturation parameters](docs/search_workflow.md).

## Use your own model

[Wrap your own model](docs/custom_models.md) ·
[Model input requirements](docs/model_input_requirements.md).
