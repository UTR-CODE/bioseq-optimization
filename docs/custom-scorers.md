# Scoring interface

For a PyTorch model, start with [SequencePredictor](pytorch-models.md). If the model requires structured inputs or preprocessing that needs its own implementation, provide an object with `score(sequences)` and `score_soft(probabilities)`. Use `task_names` to name the outputs in prediction order.

`score()` receives nucleotide strings and returns one row of model predictions per sequence. `score_soft()` receives a PyTorch tensor of shape `[batch, length, 4]` in A/C/G/T order and returns `(predictions, uncertainty)`. Use `None` for uncertainty if the model does not produce it. The predictions must retain gradients with respect to the probabilities.
