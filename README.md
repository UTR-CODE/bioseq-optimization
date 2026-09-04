# BioSeqBench Sequence Optimization Toolkit

A model-agnostic toolkit for optimizing fixed-length DNA or RNA sequences
against a user-provided activity model. The same optimizer API supports
single-task models and multi-task models such as DeepSTARR.

The reusable software lives in [`sequence_optimization/`](sequence_optimization/).
Historical dataset-specific BioSeqBench training and benchmark code remains in
[`BioSeqBench/`](BioSeqBench/), but the new optimization core does not depend on
a particular dataset or training pipeline.

## Current scope

- Input: fixed-length DNA or RNA strings.
- Model output: one activity per sequence (`[N]` or `[N, 1]`) or multiple
  activities (`[N, tasks]`).
- Initialization: user-provided seed sequences or reproducible random seeds.
- Objectives: maximize one selected output, or maximize it while penalizing
  all non-target outputs.
- Algorithms: random search, greedy ISM, simulated annealing, genetic
  algorithm, and AdaLead.
- Accounting: repeated-sequence cache and explicit model-query budget.
- Output: ranked sequences, scalar objective scores, raw per-task activities,
  uncertainty, metadata, and optimization trajectory.

One model with multiple output heads is supported. Combining multiple separate
models (for example, one model per cell type) is intentionally out of scope for
this version.

## Repository layout

```text
sequence_optimization/
├── api.py                 # scorers, sequence space, budget, result types
├── objectives.py          # single- and multi-task scalar objectives
├── algorithms.py          # optimizer registry and implementations
├── seeds.py               # seed-file and random-seed helpers
├── cli.py                 # command-line interface
└── examples/
    ├── verify_optimus_deepstarr.py
    └── models/             # checkpoint-aware model adapters

BioSeqBench/models/         # model architectures used by examples
BioSeqBench/checkpoints/    # two small checkpoints used by the smoke test
tests/                      # unit and regression tests
```

## Installation

Python 3.10 or newer is required.

### Option A: pip and a virtual environment

```bash
git clone https://github.com/UTR-CODE/bioseq-optimization.git
cd bioseq-optimization

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Option B: conda with CUDA support

```bash
conda env create -f environment.yml
conda activate bioseqbench
python -m pip install -e .
```

CPU execution is sufficient for all tests and the documented smoke run.

## Fast independent verification

First run the tests:

```bash
python -m pytest -q
```

The reference checkout currently reports:

```text
55 passed
```

Then run both real checkpoint adapters through the optimization interface:

```bash
python -m sequence_optimization.examples.verify_optimus_deepstarr
```

With Python 3.10, NumPy 2.1.2, and PyTorch 2.5.1 on CPU, the reference run
uses 302 Optimus queries and 8 DeepSTARR queries. Typical top values are:

```text
Optimus MRL:                         7.7108
DeepSTARR developmental activity:  6.7203
DeepSTARR housekeeping activity:  -0.1458
DeepSTARR specificity utility:      6.7932
```

Small floating-point differences across PyTorch versions and hardware are
normal. Query counts, shapes, task names, and ranking behavior should agree.

The two checkpoint SHA-256 hashes are:

```text
7b58549d44e33addcf88876bb7ef35fd1dfa6404083da47619ef258ea513526c  Optimus best.pt
79c070f319b0654209db3d04d3245d7ada266cd6d1515d52453f5c9a029d3127  DeepSTARR best.pt
```

## Minimal single-task example

Any callable that receives a NumPy array of strings and returns one value per
sequence can act as a model:

```python
import numpy as np

from sequence_optimization import (
    CallableScorer,
    SequenceSpace,
    genetic_algorithm_optimize,
)


def activity_model(sequences: np.ndarray) -> np.ndarray:
    return np.asarray([sequence.count("G") for sequence in sequences], dtype=float)


scorer = CallableScorer(activity_model, task_names=("activity",))
result = genetic_algorithm_optimize(
    scorer,
    SequenceSpace(length=20, alphabet="ACGT"),
    budget=10_000,
    num_candidates=100,
    seed=7,
)

print(result.to_frame().head())
result.to_frame().to_csv("designs.csv", index=False)
```

When `initial` is omitted, the toolkit constructs unique random seed sequences
from `seed`. To supply seeds explicitly:

```python
initial = np.asarray(["A" * 20, "C" * 20, "G" * 20, "T" * 20])

result = genetic_algorithm_optimize(
    scorer,
    SequenceSpace(20, "ACGT"),
    budget=1_000,
    num_candidates=4,
    initial=initial,
    seed=7,
)
```

`initial` must contain exactly `num_candidates` unique, valid sequences. A
one-sequence-per-line file can be loaded with `load_seed_sequences()`.

For RNA, use an RNA alphabet and ensure the model encoder expects `U`:

```python
rna_space = SequenceSpace(length=40, alphabet="ACGU")
```

## Multi-task objectives

Give output columns stable names when constructing the scorer:

```python
scorer = CallableScorer(
    multi_task_model,
    task_names=("developmental", "housekeeping"),
)
```

To maximize only one output and ignore all others:

```python
from sequence_optimization import TargetObjective

objective = TargetObjective(
    target="developmental",
    mode="maximize",
)
```

For CRESted-style target specificity:

```python
objective = TargetObjective(
    target="developmental",
    mode="specificity",
    penalty=1.0,
)
```

The specificity utility is:

```text
target - penalty × sum(weight_i × non_target_i) / number_of_tasks
```

`other_weights` can assign different penalties to individual outputs. The
model's raw predictions are never replaced by the scalar objective:

- `result.scores` contains the scalar utility used for ranking;
- `result.activities` contains the original `[candidates, tasks]` predictions;
- `result.to_frame()` creates `activity_<task_name>` columns.

## Optimus example

The included Optimus adapter loads the 50-nt, single-task checkpoint and
restores predictions from standardized training units to MRL units:

```python
from sequence_optimization import ISM_optimize, SequenceSpace
from sequence_optimization.examples.models import Optimus5PrimeExample

scorer = Optimus5PrimeExample.from_checkpoint(device="cpu")
result = ISM_optimize(
    scorer,
    SequenceSpace(50, "ACGT"),
    budget=302,
    num_candidates=2,
    seed=11,
)
```

For one ISM sweep, each DNA seed requires `length × 3 = 150` single-mutant
queries, in addition to its initial score.

## DeepSTARR example

DeepSTARR returns developmental and housekeeping enhancer activities:

```python
from sequence_optimization import (
    SequenceSpace,
    TargetObjective,
    genetic_algorithm_optimize,
)
from sequence_optimization.examples.models import DeepSTARRExample

scorer = DeepSTARRExample.from_checkpoint(device="cpu")
result = genetic_algorithm_optimize(
    scorer,
    SequenceSpace(249, "ACGT"),
    budget=256,
    num_candidates=32,
    objective=TargetObjective(
        target="developmental",
        mode="specificity",  # use "maximize" to ignore housekeeping
    ),
    seed=1,
)
```

## Available optimizer functions

| Function | Registry name | Summary |
|---|---|---|
| `random_optimize` | `random` | Uniform random screening |
| `ISM_optimize` / `ism_optimize` | `ism_greedy` | Exhaustive single-mutant greedy search |
| `simulated_annealing_optimize` | `simulated_annealing` | Temperature-controlled mutation acceptance |
| `genetic_algorithm_optimize` | `genetic_algorithm` | Elite selection, crossover, and mutation |
| `adalead_optimize` | `adalead` | Elite-led local adaptive search |

All helpers accept the same core arguments: `scorer`, `space`, `budget`,
`num_candidates`, `initial`, `objective`, and `seed`. Algorithm-specific
parameters such as `mutation_rate` and `elite_fraction` are also available.

Use `run()` for a registry name or `run_many()` to compare several algorithms
with independent caches and budgets.

## Wrapping a PyTorch model

For a PyTorch model, provide an encoder that converts strings into the model's
expected tensor shape:

```python
from sequence_optimization import TorchSequenceScorer

scorer = TorchSequenceScorer(
    model,
    encoder,
    device="cuda",
    batch_size=1024,
    task_names=("cell_a", "cell_b"),
)
```

The optimizer remains a black-box consumer of `score(sequences)`; it does not
depend on the model architecture or training framework.

## Command-line interface

Expose a Python function such as `my_model:predict`, then run:

```bash
python -m sequence_optimization.cli \
  --scorer my_model:predict \
  --algorithm genetic_algorithm \
  --length 249 \
  --alphabet ACGT \
  --budget 10000 \
  --num-candidates 100 \
  --task-names developmental,housekeeping \
  --target developmental \
  --objective specificity \
  --seed 7 \
  --output designs.csv
```

Add `--seed-file seeds.txt` to replace random initialization. The CLI writes
the ranked CSV plus a JSON sidecar containing the objective and run metadata.

## Reproducibility and limitations

- A query is one newly scored unique sequence by default. Duplicate sequences
  are cached and do not consume additional budget.
- Random initialization and optimizer stochasticity are controlled by `seed`.
- Current algorithms optimize fixed-length discrete sequences.
- Constraints such as forbidden motifs, fixed positions, GC windows, and
  distance from training data are not yet part of the public interface.
- Gradient-based methods such as Fast SeqProp, CbAS, and DEN have not yet been
  migrated into this standalone API.
- Historical code under `BioSeqBench/optimization/` is retained for
  reproducibility; new applications should import from `sequence_optimization`.

## Development checks

```bash
python -m py_compile sequence_optimization/*.py sequence_optimization/examples/models/*.py
python -m pytest -q
```

Before publishing this repository publicly, select and add an explicit open
source license. No license is implied merely by making a GitHub repository
public.
