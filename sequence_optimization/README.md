# `sequence_optimization`

This package contains model-agnostic DNA/RNA optimization algorithms. Models
provide sequence predictions; ready-to-use implementations and checkpoints are
provided separately by
[`bioseq-models`](https://github.com/UTR-CODE/bioseq-models).

The main workflow is:

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize, random_seed_sequences

model = load_model("optimus5prime")
initial = random_seed_sequences(length=50, count=16, seed=7)
result = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],  # or [0.6] to stop after reaching score 0.6
    num_candidates=16,
    budget=1000,
    seed=7,
)
```

Two complete runnable examples are included:

- [`examples/custom_cnn.py`](examples/custom_cnn.py) defines and optimizes a
  minimal one-output CNN.
- [`examples/external_model.py`](examples/external_model.py) loads and optimizes
  Optimus 5-Prime from `bioseq-models`.

For variable biological lengths, configure the scorer with
`pad_to_length=<fixed model length>` and `padding_side="left"` or `"right"`.
Padding uses `N`; differentiable soft inputs use equivalent all-zero vectors.
The optimized and returned sequence itself remains ACGT.

See the [project README](../README.md) for algorithm choices, multi-task
objectives, [sequence constraints](../docs/constraints.md), saturation stopping,
installation, and complete examples.
