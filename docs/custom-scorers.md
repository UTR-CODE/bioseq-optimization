# Scoring interface

Pass a custom object with `score(sequences)` and `score_soft(probabilities)` directly to MotifSift. These methods suffice for a single-output model. To refer to multiple outputs by name in an objective, add `task_names` in prediction order.

- `score()` receives a Python `list[str]`. The input alphabet is A/C/G/T/U/N. MotifSift uppercases the strings and converts U to T before calling your method. N represents an unknown base. Return a score vector of shape `[batch]` or a prediction matrix of shape `[batch, tasks]`. The default search proposes A/C/G/T sequences.
- `score_soft()` receives a PyTorch tensor of shape `[batch, length, 4]` in A/C/G/T order. Return a prediction tensor of shape `[batch]` or `[batch, tasks]` that retains gradients with respect to the input. You can also return `(predictions, uncertainty)`.

See the [complete scoring example](../examples/custom-scoring-methods.ipynb). For a PyTorch model that exposes `forward()`, use [SequencePredictor](pytorch-models.md) to supply both scoring methods.
