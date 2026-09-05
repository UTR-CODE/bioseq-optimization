# Biosequence Optimization

This package optimizes fixed-length DNA or RNA sequences using a user-provided
activity model. A model may have one output or several outputs. The optimizer
can start from user-provided sequences or generate random starting sequences.

Internally, all nucleotide sequences use the alphabet `ACGT`. Lowercase input
is converted to uppercase, and RNA `U` is automatically converted to `T`
before validation and model prediction.

## Installation

### Conda

The repository provides a complete [`environment.yml`](environment.yml):

```bash
git clone https://github.com/UTR-CODE/bioseq-optimization.git
cd bioseq-optimization
conda env create -f environment.yml
conda activate bioseq-optimization
```

### Python and pip

The same dependencies are listed in [`requirements.txt`](requirements.txt):

```bash
git clone https://github.com/UTR-CODE/bioseq-optimization.git
cd bioseq-optimization
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Ready-to-use models

Five trained checkpoints are included in the repository and can be loaded by
name. No dataset download or training step is required.

```python
from sequence_optimization import load_model

model = load_model("optimus5prime", device="cpu")
```

| Model name | Input length | Output activity | Gradient optimization |
|---|---:|---|---|
| `optimus5prime` | 50 nt | Mean ribosome load | Yes |
| `deepstarr` | 249 bp | Developmental, housekeeping | Yes |
| `dream_legnet` | 110 bp | Yeast promoter activity | No |
| `zhao_ptrc` | 74 bp | log10 fluorescence/OD600 | Yes |
| `zhou_l3` | 29 bp | Promoter activity | Yes |

Use `available_models()` to list model names and `model_info(name)` to inspect
the required sequence length and output names. A different checkpoint of the
same architecture can be supplied with `load_model(name, checkpoint=...)`.

## Available algorithms

All algorithms use the same sequence, objective, initialization, and result
interfaces.

| Python function | Method | Model requirement |
|---|---|---|
| `random_optimize` | Random screening | Black-box activity model |
| `ISM_optimize` | Greedy in-silico mutagenesis | Black-box activity model |
| `simulated_annealing_optimize` | Simulated annealing | Black-box activity model |
| `genetic_algorithm_optimize` | Genetic algorithm | Black-box activity model |
| `adalead_optimize` | AdaLead | Black-box activity model |
| `beam_search_optimize` | Beam search over single mutants | Black-box activity model |
| `cross_entropy_optimize` | Position-wise cross-entropy method | Black-box activity model |
| `cbas_optimize` | CbAS with an autoregressive GRU generator | Black-box activity model |
| `vanilla_activation_maximization_optimize` | Soft-PWM activation maximization | Differentiable model |
| `fast_seqprop_optimize` | Fast SeqProp | Differentiable model |
| `den_optimize` | Deep Exploration Network | Differentiable model |

CbAS creates and trains its sequence generator internally. DEN also creates an
internal generator, but it additionally requires gradients through the
activity model. The included Optimus, DeepSTARR, Zhao Ptrc, and Zhou L3
adapters support these differentiable algorithms.

## Use your own CNN

The optimizer does not train the activity model. First train and save your
model normally. The smallest included example uses one convolution, global
average pooling, and one activity output:

```python
import torch

from sequence_optimization import SequenceSpace, genetic_algorithm_optimize
from sequence_optimization.examples.custom_cnn import (
    SimpleActivityCNN,
    wrap_custom_cnn,
)

model = SimpleActivityCNN()
checkpoint = torch.load("my_activity_cnn.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state"])

activity_model = wrap_custom_cnn(model, device="cpu")

result = genetic_algorithm_optimize(
    activity_model,
    SequenceSpace(length=50),
    num_candidates=4,
    budget=44,
    mutation_rate=0.1,
    elite_fraction=0.5,
    seed=7,
)
```

Here `44` caps the run at 44 distinct model queries: four initial sequences and
up to 40 newly scored proposals. Omitting `initial` creates reproducible random
starting sequences. To use real seed sequences, pass a Python list through
`initial=[...]`.

The complete runnable integration is in
[`sequence_optimization/examples/custom_cnn.py`](sequence_optimization/examples/custom_cnn.py).
Its `encode_sequences()` function converts strings to `[batch, 4, length]`
tensors, while `wrap_custom_cnn()` makes the same trained model compatible
with both black-box and gradient-based optimizers.

Run that example with:

```bash
python -m sequence_optimization.examples.custom_cnn my_activity_cnn.pt
```

## Optimus 5-Prime example

Optimus accepts 50-nt sequences and predicts mean ribosome load (MRL). The
checkpoint adapter loads the included trained model and restores predictions
to the original MRL scale.

The five starting sequences below are real sequences from the Optimus MPRA
test split:

```python
from sequence_optimization import ISM_optimize, SequenceSpace, load_model

model = load_model("optimus5prime", device="cpu")

initial_sequences = [
    "CCACTCGATTAACATGTTAACAACATACTCGTCCGGCCGATCAGCGGCTA",
    "CAAATCATGTGCAGCCCTGGCGACCGTACTGCGGTACAAGAAAGTATAGT",
    "GTTATACTAGAAGAAACTTGAGATTATGGAGCAGTCCGTCAAGGACGCGA",
    "CTTAGACAAAAACAACGCGCTTTCCAGTATGCGGAGCCTTGACGGTAAAC",
    "GTATCAAATCACGGCCAACCCGACGGAGTACCCCGCGTCGATGGTCCGCC",
]

result = ISM_optimize(
    model,
    SequenceSpace(length=50),
    initial=initial_sequences,
    num_candidates=5,
    budget=755,
    seed=7,
)
```

The budget `755` is not an arbitrary formatting choice. It is exactly:

```text
5 initial predictions + 5 sequences × 50 positions × 3 alternative bases
```

It therefore performs one complete ISM sweep for every starting sequence.

The returned DataFrame begins as follows:

| sequence | score | uncertainty | activity_mrl |
|---|---:|---:|---:|
| `GTTATACTAGAAGAAACTTGAGATTATTGAGCAGTCCGTCAAGGACGCGA` | 8.1850 | 0.0 | 8.1850 |
| `GTTATACTAGAAGAAACTTGAGATTTTGGAGCAGTCCGTCAAGGACGCGA` | 8.1361 | 0.0 | 8.1361 |
| `GTTATACTAGAAGAAACTTGAGATTATCGAGCAGTCCGTCAAGGACGCGA` | 8.1199 | 0.0 | 8.1199 |
| `GTTATACTAGAAGAAACTTGAGATTAAGGAGCAGTCCGTCAAGGACGCGA` | 8.1182 | 0.0 | 8.1182 |
| `GTTATACTAGAAGAAACTTGAGATTATAGAGCAGTCCGTCAAGGACGCGA` | 8.0844 | 0.0 | 8.0844 |

`score` is the scalar value used by the optimizer. `activity_mrl` is the raw
Optimus output. They are identical for this single-task model.

To use another fixed-budget method, change the optimizer function and select a
budget appropriate to that method. If there is no defensible fixed budget, use
the saturation interface below instead of choosing an arbitrary number.

## DeepSTARR multi-task example

DeepSTARR accepts 249-bp sequences and returns two activities:

1. `developmental`
2. `housekeeping`

```python
from sequence_optimization import (
    SequenceSpace,
    TargetObjective,
    genetic_algorithm_optimize,
    load_model,
)

model = load_model("deepstarr", device="cpu")

result = genetic_algorithm_optimize(
    model,
    SequenceSpace(length=249),
    num_candidates=32,
    budget=256,
    objective=TargetObjective(
        target="developmental",
        mode="specificity",
    ),
    seed=7,
)
```

Here `256` means 32 initial predictions followed by seven generations of 32
proposals each. Omit `initial` to obtain reproducible random starting
sequences.

For a multi-task model, two objective modes are available.

Maximize only the selected output and ignore the others:

```python
objective = TargetObjective(
    target="developmental",
    mode="maximize",
)
```

Maximize the selected output while penalizing non-target outputs:

```python
objective = TargetObjective(
    target="developmental",
    mode="specificity",
    penalty=1.0,
)
```

The specificity score is:

```text
target activity - penalty × weighted non-target activities / number of tasks
```

The result retains both the scalar optimization score and the original
`developmental` and `housekeeping` predictions.

## Run without choosing a fixed budget

An optimization cannot run literally forever, so unlimited mode uses an online
plateau rule plus a safety ceiling. It monitors the mean score of the best
`num_candidates` sequences and stops when that value has not improved
materially for the configured patience period.

After the plateau is established, q80 and q90 are selected retrospectively:

- q80: the earliest candidate set reaching 80% of the observed improvement;
- q90: the earliest candidate set reaching 90% of the observed improvement;
- `knee_query_count`: the geometric knee of the normalized improvement curve.

```python
from sequence_optimization import (
    SaturationConfig,
    SequenceSpace,
    TargetObjective,
    load_model,
    optimize_to_saturation,
)

model = load_model("deepstarr", device="cpu")

saturated = optimize_to_saturation(
    model,
    SequenceSpace(249),
    algorithm="genetic_algorithm",
    num_candidates=32,
    objective=TargetObjective("developmental", mode="specificity"),
    saturation=SaturationConfig(fractions=(0.8, 0.9)),
    seed=7,
)

q80_sequences = saturated[0.8].sequences
q90_sequences = saturated[0.9].sequences
knee_query_count = saturated.knee_query_count
```

If query limits are omitted, they are derived from `num_candidates`:

- begin plateau testing after `10 × num_candidates` queries;
- require `20 × num_candidates` queries without material improvement;
- use `1000 × num_candidates` as the safety ceiling.

These defaults can be replaced with `minimum_queries`,
`plateau_patience_queries`, and `maximum_queries` in `SaturationConfig`.

## RNA input

Use the same `ACGT` sequence space for DNA and RNA models:

```python
space = SequenceSpace(length=50)
```

Inputs containing `U` are accepted but normalized before the model sees them:

```text
acguuacg  →  ACGTTACG
```

Generated and returned sequences therefore always contain `T`, never `U`.

## Independent verification

Run the complete test suite:

```bash
python -m pytest -q
```

Then load all bundled checkpoints and run the Optimus/DeepSTARR optimization
smoke test:

```bash
python -m sequence_optimization.examples.verify_pretrained_models
python -m sequence_optimization.examples.verify_optimus_deepstarr
```

Small floating-point differences between PyTorch versions and CPU/GPU devices
are expected. Query counts, output shapes, task names, and ranking behavior
should agree.
