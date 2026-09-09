# Biosequence Optimization

`bioseq-optimization` is a model-agnostic DNA/RNA sequence-design toolkit.
It contains optimization algorithms, objectives, sequence initialization,
query accounting, and scorer interfaces. It does **not** contain model
architectures, training code, datasets, or checkpoints.

Ready-to-use Optimus 5-Prime, DeepSTARR, and other reproduced activity models
are maintained in the separate
[`bioseq-models`](https://github.com/UTR-CODE/bioseq-models) repository. A model
loaded there can be passed directly to every compatible optimizer here.

The current workflow optimizes one activity scorer at a time. A single-output
model is used directly. Separate models for different cell types can each be
optimized independently, but combining several model objects into one joint
optimization objective is not currently supported.

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

## Examples

### 1. A minimal custom CNN

This complete example defines a one-output CNN, wraps it as a scorer, and runs
the genetic algorithm. The fixed weights make the demonstration reproducible;
in an application, replace them with weights loaded from a trained checkpoint.

```python
import torch
from torch import nn
from torch.nn import functional as F

from sequence_optimization import (
    SequenceSpace,
    TorchSequenceScorer,
    genetic_algorithm_optimize,
)

space = SequenceSpace(length=20)


class MinimalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Conv1d(4, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.convolution.weight.zero_()
            self.convolution.weight[0, 2, 0] = 1.0  # reward G

    def forward(self, inputs):
        return self.convolution(inputs).mean(dim=2)


def encode(sequences):
    encoded = torch.as_tensor(space.encode(sequences), dtype=torch.long)
    return F.one_hot(encoded, num_classes=4).float().transpose(1, 2)


scorer = TorchSequenceScorer(
    MinimalCNN(),
    encode,
    device="cpu",
    task_names=("activity",),
)
result = genetic_algorithm_optimize(
    scorer,
    space,
    num_candidates=4,
    budget=40,
    mutation_rate=0.1,
    seed=7,
)
print(result.to_frame())
```

The runnable version is
[`sequence_optimization/examples/custom_cnn.py`](sequence_optimization/examples/custom_cnn.py).

### 2. Load Optimus 5-Prime from `bioseq-models`

This example loads Optimus from `bioseq-models`, starts from five 50-nt 5′UTR
sequences, and returns the five best unique candidates found by the genetic
algorithm.

```python
from bioseq_models import load_model
from sequence_optimization import SequenceSpace, genetic_algorithm_optimize

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
    SequenceSpace(length=50),
    initial=initial,
    num_candidates=5,
    budget=80,
    mutation_rate=0.1,
    elite_fraction=0.4,
    seed=7,
)

print(result.sequences[:5])
print(result.activities[:5])
```

No wrapper is required for Optimus with any of the eight black-box algorithms.
The runnable version is
[`sequence_optimization/examples/external_model.py`](sequence_optimization/examples/external_model.py).

`budget=80` means that this run may query the activity model for at most 80
candidate sequences. It is a model-evaluation budget, not a training epoch or
sequence count in the final result.

With the bundled checkpoint, the first five rows are:

| sequence | score | uncertainty | activity_mrl |
|---|---:|---:|---:|
| GTGAGATCAAGATAAATTTGTCCCTGTTTATCTCTAATCCACTGTCTAAA | 8.871757 | 0.000000 | 8.871757 |
| GAGAGAACTAGCTTAATTTGTCCCTGTTTACCTCTAATCTACTTTCTGAA | 8.739843 | 0.000000 | 8.739843 |
| GTGAGATCTAGATAAATTTGTCCCTGTTTACCTCTAATCCACTTCCTAAA | 8.737933 | 0.000000 | 8.737933 |
| GAGAGAACTAGATTAATTTGTCCCTGTTTACCTCTAATCGACTTTCTAAA | 8.715504 | 0.000000 | 8.715504 |
| GAGAGGACTATATTAATTTGTCGCTTTTTAACTCTAACCGACTTTCTAGA | 8.691080 | 0.000000 | 8.691080 |

The arrays are also available directly as `result.sequences`, `result.scores`,
`result.uncertainties`, `result.activities`, and `result.task_names`.

## Variable biological length with fixed model input

The sequence being designed may be shorter than the model input. Set
`pad_to_length` and choose the padding side on the scorer. The optimizer still
searches only the biological ACGT sequence; `N` is added only immediately
before model encoding.

```python
from bioseq_models import load_model
from sequence_optimization import SequenceSpace, genetic_algorithm_optimize

model = load_model(
    "optimus5prime",
    pad_to_length=50,
    padding_side="left",       # or "right"
)

result = genetic_algorithm_optimize(
    model,
    SequenceSpace(length=40),  # optimize the real 40-nt sequence
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

## Algorithms

The [32-model single-output compatibility report](docs/single_task_model_compatibility_2026-09-09.md)
records real-checkpoint runs: all 32 models pass the eight black-box algorithms;
28 also pass the three gradient algorithms with wrappers (Cetnar on CPU).
The remaining four need model-specific differentiable adapters.

All algorithms use the same scorer, sequence space, initialization, objective,
and result contract.

| Function | Algorithm | Optimus usage |
|---|---|---|
| `random_optimize` | Random search | Direct |
| `ism_optimize` / `ISM_optimize` | Greedy in-silico mutagenesis | Direct |
| `simulated_annealing_optimize` | Simulated annealing | Direct |
| `genetic_algorithm_optimize` | Genetic algorithm | Direct |
| `adalead_optimize` | AdaLead-style local search | Direct |
| `beam_search_optimize` | Beam search | Direct |
| `cross_entropy_optimize` | Cross-entropy method | Direct |
| `cbas_optimize` | CbAS with an internal autoregressive generator | Direct |
| `vanilla_activation_maximization_optimize` | Activation maximization | Differentiable wrapper |
| `fast_seqprop_optimize` | Fast SeqProp | Differentiable wrapper |
| `den_optimize` | Deep exploration network | Differentiable wrapper |

CbAS creates and fits its own proposal generator from the seed sequences and
queried activities. DEN also creates an internal generator, but it optimizes
that generator through the activity model, so the supplied scorer must support
`score_soft`.

All eleven registered algorithms are exercised against the real Optimus
checkpoint in
[`tests/test_bioseq_models_integration.py`](tests/test_bioseq_models_integration.py).
The eight black-box algorithms accept the loaded model directly. For the three
gradient algorithms, expose the differentiable PyTorch network through the
existing wrapper:

```python
from sequence_optimization import DifferentiableTorchSequenceScorer

differentiable_optimus = DifferentiableTorchSequenceScorer(
    model.model,
    model.encoder,
    device=model.device,
    task_names=model.task_names,
    soft_input_transform=lambda probabilities: probabilities.transpose(1, 2),
)
```

The transpose adapts the optimizer's canonical `[batch, length, A/C/G/T]`
representation to Optimus's `[batch, A/C/G/T, length]` tensor layout. Pass
`differentiable_optimus` to any of the three gradient optimizers.

Use `run("algorithm_name", ...)` for the registry-based interface, or
`run_many(...)` to compare several algorithms with independent query budgets.

## Seed sequences or random initialization

Pass `initial=[...]` to any optimizer to use known sequences. If `initial` is
omitted, candidates are sampled uniformly from `SequenceSpace`. The helper
`load_seed_sequences(path, space)` reads one sequence per line; blank lines and
lines starting with `#` are ignored.

## Stop at a plateau instead of choosing a fixed budget

`optimize_to_saturation` tracks the mean score of the retained candidates,
stops after a plateau, and returns the earliest candidate sets reaching chosen
fractions of the observed improvement.

```python
from sequence_optimization import SaturationConfig, optimize_to_saturation

saturated = optimize_to_saturation(
    model,
    SequenceSpace(length=50),
    algorithm="genetic_algorithm",
    num_candidates=16,
    initial=initial,
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

Wrap a callable with `CallableScorer`, a frozen PyTorch model with
`TorchSequenceScorer`, or a differentiable PyTorch model with
`DifferentiableTorchSequenceScorer`. See the local
[`custom_cnn.py`](sequence_optimization/examples/custom_cnn.py) example for a
complete minimal integration.

A scorer must accept a batch of sequence strings and return either `[batch]`
or `[batch, tasks]` activities. The optimizer never trains the supplied
activity model.

## Command line

The CLI accepts a Python callable in `package.module:function` form:

```bash
python -m sequence_optimization.cli \
  --scorer my_package.my_module:predict_activity \
  --algorithm genetic_algorithm \
  --length 40 \
  --pad-to-length 50 \
  --padding-side left \
  --budget 160 \
  --num-candidates 16 \
  --output candidates.csv
```

Use `--budget unlimited` to enable plateau stopping and
`--saturation-fractions 0.8,0.9` to retain q80/q90 selections.

## License and model provenance

This repository contains only optimization software. Consult `bioseq-models`
for checkpoint provenance, model-specific dependencies, and model licensing.
