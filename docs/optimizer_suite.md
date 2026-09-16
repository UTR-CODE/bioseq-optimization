# Run selected optimizers together

`optimize_algorithms` runs each selected optimizer independently from the same
initial population and returns an `OptimizerSuiteResult`. The initial
population may be larger than `num_candidates`; every optimizer uses the full
initial population and retains exactly `num_candidates` results. Raw model
predictions are cached across optimizers by default, so the shared initial
population and any overlapping proposals require only one model evaluation.

```python
from bioseq_models import load_model
from sequence_optimization import (
    SequencePredictor,
    optimize_algorithms,
    random_seed_sequences,
)

model = load_model(
    "optimus5prime",
    device="cuda",
    batch_size=4096,
)
gradient_model = SequencePredictor(
    model.model,
    device=model.device,
    task_names=model.task_names,
)
initial = random_seed_sequences(length=50, count=1000, seed=7)

suite = optimize_algorithms(
    model,
    algorithms=[
        "random",
        "genetic_algorithm",
        "basesearch",
        "cross_entropy_method",
        "cbas",
        "fast_seqprop",
    ],
    differentiable_model=gradient_model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=100,
    budget=10_000,
    seed=7,
)

all_candidates = suite.to_frame()
summary = suite.summary_frame()
suite.save_csv("optimus_mrl_suite")
```

`to_frame()` includes the optimizer name, within-optimizer rank, objective
score, MRL prediction, optimizer query count, actual model query count, and
elapsed time. `summary_frame()` also reports `max_discrete_batch_size` for checking
whether discrete model calls are large enough to use the accelerator
efficiently. `soft_query_count` and `max_soft_batch_size` report the additional
forward/backward work performed by gradient optimizers. The summary also
contains candidate count, throughput, best and mean score, and stop reason.
`save_csv()` writes both tables, the initial sequences, and one result file per
optimizer. Set `share_model_cache=False` when model predictions should be
recomputed independently for every optimizer.

The supported names are returned by `available_algorithms()`. The gradient
optimizers `vanilla_activation_maximization`, `fast_seqprop`, `den`, and
`ledidi` require `differentiable_model`; black-box algorithms use `model`.
Use `optimizer_kwargs` for per-algorithm settings:

```python
suite = optimize_algorithms(
    model,
    algorithms=["basesearch", "cbas", "cross_entropy_method"],
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=100,
    budget=10_000,
    optimizer_kwargs={
        "basesearch": {"basesearch_batch_size": 1000},
        "cbas": {"cbas_batch_size": 1000},
        "cross_entropy_method": {"cross_entropy_batch_size": 1000},
    },
)
```

When omitted, the BaseSearch, CbAS, and cross-entropy batch sizes use the
smaller of the initial population size and the predictor's configured inference
`batch_size`. Adjust their individual settings to trade optimizer update
frequency against accelerator throughput and memory.

The runnable Optimus command exposes algorithm selection and output paths:

```bash
python -m sequence_optimization.examples.optimus_single_task_suite \
  --algorithms basesearch genetic_algorithm random \
  --initial-count 1000 \
  --num-candidates 100 \
  --budget 10000 \
  --device cuda \
  --output-dir optimus_mrl_suite
```
