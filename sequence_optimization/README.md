# `sequence_optimization`

This package contains model-agnostic DNA/RNA optimization algorithms. Models
are external scorer objects; ready-to-use implementations and checkpoints are
provided separately by
[`bioseq-models`](https://github.com/UTR-CODE/bioseq-models).

The main workflow is:

```python
from bioseq_models import load_model
from sequence_optimization import SequenceSpace, genetic_algorithm_optimize

model = load_model("optimus5prime")
result = genetic_algorithm_optimize(
    model,
    SequenceSpace(length=50),
    num_candidates=16,
    budget=160,
    seed=7,
)
```

Two complete runnable examples are included:

- [`examples/custom_cnn.py`](examples/custom_cnn.py) defines and optimizes a
  minimal one-output CNN.
- [`examples/external_model.py`](examples/external_model.py) loads Optimus
  5-Prime from `bioseq-models` and passes it directly to a black-box optimizer.

For variable biological lengths, configure the scorer with
`pad_to_length=<fixed model length>` and `padding_side="left"` or `"right"`.
Padding uses `N`; differentiable soft inputs use equivalent all-zero vectors.
The optimized and returned sequence itself remains ACGT.

See the [project README](../README.md) for algorithm choices, multi-task
objectives, saturation stopping, installation, and complete examples.
