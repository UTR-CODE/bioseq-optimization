# Biosequence Optimization

`bioseq-optimization` is a model-agnostic DNA/RNA sequence-design toolkit.
It contains optimization algorithms, objectives, sequence initialization,
query accounting, and scorer interfaces. It does **not** contain model
architectures, training code, datasets, or checkpoints.

Ready-to-use Optimus 5-Prime, DeepSTARR, and other reproduced activity models
are maintained in the separate
[`bioseq-models`](https://github.com/UTR-CODE/bioseq-models) repository.

The workflow accepts single-output models (Optimus), multi-output models
(DeepSTARR), or independent models combined with `CombinedScorer`.
A direction list selects which outputs to raise, lower, or ignore.

## Installation

Install only the optimization package:

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-optimization.git
```

To use the ready-to-use models as well:

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-models.git
```

For development:

```bash
git clone git@github.com:UTR-CODE/bioseq-optimization.git
cd bioseq-optimization
python -m pip install -e .
python -m pytest -q
```

## Algorithms

| Function | Algorithm | Scorer requirement |
|---|---|---|
| `random_optimize` | Random search | `score` |
| `basesearch_optimize` | BaseSearch priority-decayed mutation search | `score` |
| `ism_optimize` / `ISM_optimize` | Greedy in-silico mutagenesis | `score` |
| `simulated_annealing_optimize` | Simulated annealing | `score` |
| `genetic_algorithm_optimize` | Genetic algorithm | `score` |
| `adalead_optimize` | AdaLead-style local search | `score` |
| `beam_search_optimize` | Beam search | `score` |
| `cross_entropy_optimize` | Cross-entropy method | `score` |
| `cbas_optimize` | CbAS with an internal autoregressive generator | `score` |
| `vanilla_activation_maximization_optimize` | Activation maximization | `score_soft` |
| `fast_seqprop_optimize` | Fast SeqProp | `score_soft` |
| `den_optimize` | Deep exploration network | `score_soft` |
| `ledidi_optimize` | Ledidi-style objective maximization | `score_soft` |

[Algorithm parameters](docs/algorithms.md) · [Model input requirements](docs/model_input_requirements.md)

## Examples

### Single task: Optimus 5-Prime from `bioseq-models`

This example loads Optimus from `bioseq-models`, starts from five 50-nt 5′UTR
sequences, and returns the five best unique candidates found by the genetic
algorithm.

`model.score(sequences)` accepts nucleotide strings and returns a NumPy
prediction array; the loaded model handles its input conversion.

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize

model = load_model("optimus5prime", device="cpu")

initial = [
    "CCACTCGATTAACATGTTAACAACATACTCGTCCGGCCGATCAGCGGCTA",
    "CAAATCATGTGCAGCCCTGGCGACCGTACTGCGGTACAAGAAAGTATAGT",
    "GTTATACTAGAAGAAACTTGAGATTATGGAGCAGTCCGTCAAGGACGCGA",
    "CTTAGACAAAAACAACGCGCTTTCCAGTATGCGGAGCCTTGACGGTAAAC",
    "GGTACCTCTAGTTTAATCCCTCCCTTATACGGTGCAACCCGCTCCCCTCA",
]

result = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=5,
    budget=80,
    mutation_rate=0.1,
    elite_fraction=0.4,
    seed=7,
)

print(result.sequences[:5])
print(result.activities[:5])
```

`budget=80` means that this run may query the activity model for at most 80
candidate sequences. It is a model-evaluation budget, not a training epoch or
sequence count in the final result.

The arrays are available as `result.sequences`, `result.scores`,
`result.uncertainties`, `result.activities`, and `result.task_names`.

### Multi-task: three-output CNN

The CNN below predicts three values per sequence. Its demonstration weights
measure G, C and T fractions centered at 0.25; use trained weights for a real task.

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


predictor = SequencePredictor(
    ThreeOutputCNN(),
    task_names=["output_1", "output_2", "output_3"],
    device="cpu",
)
result = genetic_algorithm_optimize(
    predictor,
    length=20,
    objective=["maximize", "minimize", "ignore"],
    num_candidates=4,
    budget=80,
    mutation_rate=0.1,
    seed=7,
)
print(result.to_frame())
```

`objective` follows the output column order: `"maximize"` raises an output,
`"minimize"` lowers it, and `"ignore"` excludes it from optimization.
Use `["maximize", "maximize", "maximize"]` to raise all three.

The default objective maximizes the minimum signed prediction across active
outputs. All raw predictions, including ignored outputs, remain in
`result.activities`. For fixed output scales or thresholds, use
`MultiTaskObjective(directions=[...], scales=[...], ...)`.

[Three-output CNN example](sequence_optimization/examples/multitask_cnn.py).

### Multiple models: lentiMPRA

Combine single-output scorers in output order. Raise K562 while lowering HepG2
and WTC11:

```python
from bioseq_models import load_model
from sequence_optimization import CombinedScorer, genetic_algorithm_optimize

scorer = CombinedScorer(
    [load_model(name, device="cpu") for name in
     ["agarwal_k562", "agarwal_hepg2", "agarwal_wtc11"]],
    names=["K562", "HepG2", "WTC11"],
)
result = genetic_algorithm_optimize(
    scorer, length=230,
    objective=["maximize", "minimize", "minimize"],
    num_candidates=4, budget=100, seed=7,
)
print(result.to_frame())
```

### Multiple models: DeepSTARR Mouse

Use the same optimizer to raise heart activity and lower limb and midbrain
activity. These are tissue models:

```python
scorer = CombinedScorer(
    [load_model(name, device="cpu") for name in
     ["deepstarr_mouse_heart_activity", "deepstarr_mouse_limb_activity",
      "deepstarr_mouse_midbrain_activity"]],
    names=["heart", "limb", "midbrain"],
)
result = genetic_algorithm_optimize(
    scorer, length=1001,
    objective=["maximize", "minimize", "minimize"],
    num_candidates=4, budget=100, seed=7,
)
print(result.to_frame())
```

[Python example functions](sequence_optimization/examples/multiple_models.py).
Each child scorer must accept the
same designed sequence and handle its own encoding, padding or fixed context.
Gradient algorithms require every child to expose a differentiable `score_soft`;
wrap models individually before combining them.

`budget=100` counts combined candidate evaluations: each of three models gets
up to 100 sequence evaluations (300 total, excluding additional gradient
evaluations and internal ensemble members). Cumulative counts are available as
`scorer.model_query_counts` and `scorer.model_soft_query_counts`.
Cache hits can consume proposal budget without another model forward pass.
All raw outputs remain in the result. To adjust different output scales, use
fixed reference scales with `MultiTaskObjective(directions=[...], scales=[...])`.
These examples optimize raw predictions; search directions do not guarantee
absolute activity levels.

## Variable biological length

The sequence being designed may be shorter than the model input. Set
`pad_to_length` and choose the padding side on the scorer. The optimizer still
searches only the biological ACGT sequence; `N` is added only immediately
before model encoding.

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize

model = load_model(
    "optimus5prime",
    pad_to_length=50,
    padding_side="left",       # or "right"
)

result = genetic_algorithm_optimize(
    model,
    length=40,  # optimize the real 40-nt sequence
    num_candidates=16,
    budget=160,
    seed=7,
)
```

For discrete scoring, padding produces strings such as
`NNNNNNNNNNACGT...`. Model encoders should encode `N` as an all-zero
nucleotide vector. For differentiable optimization the toolkit adds the same
all-zero vectors to the soft one-hot tensor, so left/right padding behaves the
same for black-box and gradient algorithms.

RNA input is accepted, but `U` is always normalized to `T`. Internally and in
all returned candidates the alphabet is ACGT, never ACGU.

## Seed sequences or random initialization

Pass `initial=[...]` to any optimizer to use known sequences. If `initial` is
omitted, candidates are sampled uniformly at the requested length. The helper
`load_seed_sequences(path, length=50)` reads one sequence per line; blank lines and
lines starting with `#` are ignored.

## Stop at a plateau instead of choosing a fixed budget

`optimize_to_saturation` tracks the mean score of the retained candidates,
stops after a plateau, and returns the earliest candidate sets reaching chosen
fractions of the observed improvement.

```python
from sequence_optimization import SaturationConfig, optimize_to_saturation

saturated = optimize_to_saturation(
    model,
    length=50,
    algorithm="genetic_algorithm",
    num_candidates=16,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9),
        maximum_queries=800,
    ),
    seed=7,
)

q80 = saturated[0.8]
q90 = saturated[0.9]
```

`maximum_queries` is only a safety ceiling. The normal stop condition is the
detected plateau. Here q80 and q90 mean 80% and 90% of the improvement from the
first complete candidate archive to the final observed plateau.

## Use your own model

See [Wrap your own model](docs/custom_models.md) for input conversion and
prediction examples.

## License and model provenance

This repository contains only optimization software. Consult `bioseq-models`
for checkpoint provenance, model-specific dependencies, and model licensing.
