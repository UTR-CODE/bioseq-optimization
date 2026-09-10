# Wrap your own model

An optimizer calls `predictor.score(sequences)` with a batch of nucleotide
strings. `SequencePredictor` wraps an existing PyTorch model and handles input
conversion, device placement, evaluation mode, and batched inference. It does
not initialize new model weights or train the model.

## Default CNN input

For a model accepting float32 `[N, 4, L]` tensors in A/C/G/T channel order:

```python
from sequence_optimization import SequencePredictor, basesearch_optimize

predictor = SequencePredictor(model, device="cpu")  # your existing nn.Module
predictions = predictor.score(["ACGT" * 5]).scores
result = basesearch_optimize(
    predictor, length=20, objective=["maximize"],
    num_candidates=4, budget=40, seed=7,
)
```

[Complete minimal CNN example](../sequence_optimization/examples/custom_cnn.py).
Load trained weights into your model before constructing the predictor.

## Custom channel order and layout

This model expects `[N, L, 4]` in A/G/C/T order:

```python
from functools import partial
import torch
from sequence_optimization import SequencePredictor, basesearch_optimize, onehot


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.readout.weight.copy_(torch.tensor([[0., 1., 0., 0.]]))

    def forward(self, values):
        return self.readout(values.mean(dim=1))


model = MyModel()  # example weights reward G; use trained weights in applications
predictor = SequencePredictor(
    model,
    onehot=partial(onehot, alphabet="AGCT", channels_first=False),
    task_names=["activity"],
    device="cpu",
    batch_size=256,
)
print(predictor.score(["AGGT", "AAAA"]).scores)

result = basesearch_optimize(
    predictor, length=20, objective=["maximize"],
    num_candidates=4, budget=40, seed=7,
)
print(result.to_frame())
```

`onehot=` accepts a callable taking a batch of strings and returning a NumPy
array or PyTorch tensor. Its channel order, layout, dtype, and handling of
unknown bases must match model training. The model returns `[N]` or `[N, tasks]`.
`predictor.score()` returns a `ScoreBatch`; `.scores` contains the predictions.

## Gradient optimization

Use `DifferentiableSequencePredictor` when optimizing through the model's
gradients. For the A/G/C/T, channels-last model above:

```python
from sequence_optimization import DifferentiableSequencePredictor, ledidi_optimize

predictor = DifferentiableSequencePredictor(
    model,
    onehot=partial(onehot, alphabet="AGCT", channels_first=False),
    soft_input_transform=lambda values: values[:, :, [0, 2, 1, 3]],
    task_names=["activity"],
    device="cpu",
)
result = ledidi_optimize(
    predictor, length=20, objective=["maximize"],
    num_candidates=4, budget=40, seed=7,
)
```

The optimizer supplies soft `[N, L, A/C/G/T]` tensors. `soft_input_transform`
must match the string conversion while preserving gradients; do not convert
these tensors to NumPy or discretize them. Model parameters are frozen while
gradients flow to sequence inputs.

[Custom model example functions](../sequence_optimization/examples/custom_model.py).

## Models loaded from bioseq-models

`load_model` returns a predictor that already accepts strings:

```python
from bioseq_models import load_model

model = load_model("optimus5prime", device="cpu")
predictions = model.score(["ACGT" * 12 + "AC"])  # NumPy [1, 1]
```

`.score()` performs the model's input conversion. `model.model` is the underlying
PyTorch network and expects the model-specific tensor input.
