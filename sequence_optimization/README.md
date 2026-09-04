# Sequence Optimization Toolkit

This directory is the standalone optimization software. It deliberately does
not contain datasets, training loops, checkpoints, or dataset-specific scoring
logic.

## User workflow

1. Describe the sequence space with `SequenceSpace(length, alphabet)`.
2. Wrap a model as a `SequenceScorer` using `CallableScorer` or
   `TorchSequenceScorer`. A model may return `[N]`, `[N, 1]`, or `[N, tasks]`
   activity values.
3. Choose seeds: pass exactly `num_candidates` strings, load them with
   `load_seed_sequences`, or omit them for reproducible random seeds.
4. Select an objective and call `optimize`, a named helper such as
   `ISM_optimize`, or the lower-level `run`/`run_many` API.

```python
from sequence_optimization import (
    CallableScorer, OptimizationConfig, SequenceSpace, run,
)

space = SequenceSpace(length=20, alphabet="ACGT")
scorer = CallableScorer(lambda seqs: [sequence.count("G") for sequence in seqs])
config = OptimizationConfig(budget=10_000, num_candidates=100)
result = run("genetic_algorithm", scorer, space, config, seed=7)
result.to_frame().to_csv("designs.csv", index=False)
```

The result's `score` is the scalar optimization utility. Raw model outputs are
retained in `activities`, labeled by `task_names`, and exported by `to_frame()`
as `activity_<task>` columns.

## Multi-task objectives

`TargetObjective(mode="maximize")` ignores non-target outputs. The
CRESted-style `mode="specificity"` rewards the target and penalizes all other
outputs:

`utility = target - penalty * sum(weight_i * other_i) / number_of_tasks`

```python
import numpy as np
from sequence_optimization import (
    CallableScorer, SequenceSpace, TargetObjective,
    genetic_algorithm_optimize,
)

def model(sequences):
    # Any real model may be used here; rows are sequences and columns are tasks.
    return np.asarray([[s.count("G"), s.count("C")] for s in sequences])

scorer = CallableScorer(model, task_names=("cell_a", "cell_b"))
result = genetic_algorithm_optimize(
    scorer,
    SequenceSpace(20),
    budget=10_000,
    num_candidates=100,
    objective=TargetObjective(target="cell_a", mode="specificity"),
    seed=7,
)
```

Use `mode="maximize"` with the same named target when the other outputs should
not affect selection. `other_weights` and `penalty` can tune the specificity
tradeoff.

## Included algorithms

`random`, `ism_greedy`, `simulated_annealing`, `genetic_algorithm`, and
`adalead` share the same black-box interface. Their direct helpers are
`random_optimize`, `ISM_optimize`/`ism_optimize`,
`simulated_annealing_optimize`, `genetic_algorithm_optimize`, and
`adalead_optimize`. Algorithms requiring a
differentiable model (categorical gradient ascent, Fast SeqProp, DEN) or a
learned proposal distribution (CbAS) can be added through
`register_algorithm` with a specialized scorer protocol; they do not require
changes to the core API.

## Optimus and DeepSTARR checkpoint examples

The optional example wrappers load the checkpoints already collected in this
repository. Optimus is a single-task MRL model:

```python
from sequence_optimization import ISM_optimize, SequenceSpace
from sequence_optimization.examples.models import Optimus5PrimeExample

scorer = Optimus5PrimeExample.from_checkpoint(device="cpu")
result = ISM_optimize(
    scorer,
    SequenceSpace(50),
    budget=604,             # 4 seeds + one 150-mutant ISM neighborhood each
    num_candidates=4,
    seed=1,                 # omit initial=... to generate reproducible seeds
)
```

DeepSTARR exposes developmental and housekeeping outputs:

```python
from sequence_optimization import (
    SequenceSpace, TargetObjective, genetic_algorithm_optimize,
)
from sequence_optimization.examples.models import DeepSTARRExample

scorer = DeepSTARRExample.from_checkpoint(device="cpu")
result = genetic_algorithm_optimize(
    scorer,
    SequenceSpace(249),
    budget=256,
    num_candidates=32,
    objective=TargetObjective(
        target="developmental",
        mode="specificity",  # or "maximize" to ignore housekeeping
    ),
    seed=1,
)
```

This interface intentionally handles one model with one or more output heads.
Combining separate per-cell-type models is reserved for a later ensemble-level
optimizer.

## Command line

For a callable `package.module:function`, multi-task selection is available as:

```bash
python -m sequence_optimization.cli \
  --scorer package.module:predict --algorithm genetic_algorithm \
  --length 249 --budget 10000 --num-candidates 100 \
  --task-names developmental,housekeeping \
  --target developmental --objective specificity --output designs.csv
```

## Examples

`examples/models/` catalogs model implementations collected in the original
BioSeqBench project (WeakSeqCNN, Optimus, TFBind8, DREAM LegNet, DeepSTARR,
PARM, lentiMPRA, Kosuri, Zhao, and Zhou). These are lazy references/adapters,
not copies of the model implementations, so the data/model packages remain
independent and cannot silently diverge.
