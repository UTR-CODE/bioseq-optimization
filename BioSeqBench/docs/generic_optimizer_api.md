# Generic sequence optimizer API

The project now separates sequence optimization from datasets and model
architectures. A model only needs to implement one method:

```python
class MyScorer:
    def score(self, sequences) -> ScoreBatch:
        # sequences is a 1-D array/list of strings
        return ScoreBatch(scores, uncertainties)
```

The search code uses a `SequenceSpace` for alphabet/length validation and an
`OptimizationConfig` for budget and algorithm hyperparameters:

```python
from BioSeqBench.optimization import (
    TorchSequenceScorer, OptimizationConfig, SequenceSpace, run_optimizer,
)

space = SequenceSpace(length=50, alphabet="ACGT")
scorer = TorchSequenceScorer(
    model=my_model,
    encoder=lambda sequences: encode_for_my_model(sequences),
    device="cuda",
)
config = OptimizationConfig(budget=50_000, num_candidates=100)
result = run_optimizer(
    "ism_greedy", initial_sequences, scorer, space, config, seed=7
)
```

The common result object contains `sequences`, `scores`, `uncertainties`,
`query_count`, `metadata`, and an optional search `trajectory`. Results can be
converted to a dataframe with `result.to_frame()`.

Built-in black-box strategies are registered under:

```text
random
ism_greedy
simulated_annealing
genetic_algorithm
adalead
```

A new optimizer is a function with the same five arguments
`(initial, scorer, space, config, rng)` and can be registered without touching
any model or dataset code:

```python
@register_optimizer("my_method")
def my_method(initial, scorer, space, config, rng):
    ...
```

`CachedScorer` adds a maximum query budget and caches duplicate sequences. The
existing TFBind8, 5′UTR, and DREAM benchmark modules remain available for
reproducibility; this core API is the recommended interface for new datasets,
strong/weak model wrappers, and user-selected algorithm collections.
