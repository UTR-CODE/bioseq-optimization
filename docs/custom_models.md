# Use a custom PyTorch model

`SequencePredictor` accepts a PyTorch model whose weights have already been
loaded. The default input is a float32 A/C/G/T one-hot tensor with shape
`[batch, 4, sequence_length]`. The same predictor works with every discrete and
gradient optimizer.

## Default input

```python
import torch

from sequence_optimization import (
    SequencePredictor,
    genetic_algorithm_optimize,
    random_seed_sequences,
)


class ActivityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(4, 32, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, values):
        return self.network(values)


model = ActivityModel()
model.load_state_dict(torch.load("activity_model.pt", weights_only=True))
predictor = SequencePredictor(model, device="cpu")

initial = random_seed_sequences(length=200, count=4, seed=7)
result = genetic_algorithm_optimize(
    predictor,
    length=200,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
)
print(result.to_frame())
```

The model may return `[batch]` or `[batch, tasks]`. It may instead return
`(activities, uncertainties)` with matching shapes. Supply `task_names` to
`SequencePredictor` when named output columns are useful.

## Custom input

Subclass `SequencePredictor` and override `prepare_input`. This method receives
canonical A/C/G/T values with shape `[batch, sequence_length, 4]` for both
discrete and gradient optimization. Return the tensor layout expected by the
model, including an input shaped `[batch, N, sequence_length]`.

The following model expects five channels: the four nucleotide channels plus a
GC indicator channel.

```python
import torch

from sequence_optimization import SequencePredictor


class FiveChannelPredictor(SequencePredictor):
    def prepare_input(self, values):
        nucleotide_channels = values.transpose(1, 2)
        gc_channel = values[:, :, 1:3].sum(dim=2).unsqueeze(1)
        return torch.cat((nucleotide_channels, gc_channel), dim=1)


model = FiveChannelModel()
model.load_state_dict(torch.load("five_channel_model.pt", weights_only=True))
predictor = FiveChannelPredictor(model, device="cuda")
```

Use PyTorch tensor operations in `prepare_input`. Converting `values` to NumPy,
detaching it, or taking a hard argmax prevents gradient optimizers from
differentiating through the sequence. Discrete optimizers call `score`, while
gradient optimizers call `score_soft`; both paths use `prepare_input`.

For a channels-last model, return `values` directly. To change channel order,
index the final dimension before transposing:

```python
class AGCTPredictor(SequencePredictor):
    def prepare_input(self, values):
        return values[:, :, [0, 2, 1, 3]].transpose(1, 2)
```

### Non-standard input shapes

`prepare_input` can also add dimensions or derive a differentiable feature
representation. It receives the canonical tensor before every model call, so
the same conversion is applied to both `score` and `score_soft`.

For example, a model that expects `[batch, 4, sequence_length, 1]` can add a
singleton dimension while preserving the sequence gradients:

```python
class FourDimensionalPredictor(SequencePredictor):
    def prepare_input(self, values):
        return values.transpose(1, 2).unsqueeze(-1)
```

A model that expects a learned vector at each position can use a weighted
embedding instead of converting the probabilities to integer token IDs:

```python
class EmbeddedPredictor(SequencePredictor):
    def prepare_input(self, values):
        # values: [batch, sequence_length, 4]
        # embedding.weight: [4, embedding_size]
        return values @ self.model.embedding.weight
```

The model in this example should accept `[batch, sequence_length,
embedding_size]`. Matrix multiplication keeps `score_soft` differentiable;
`argmax`, integer casts, and NumPy conversions do not.

If a model only accepts integer token IDs (for example, `[batch,
sequence_length]`), mark the adapter as discrete-only and select only
black-box algorithms:

```python
class TokenIdPredictor(SequencePredictor):
    supports_soft_scoring = False

    def prepare_input(self, values):
        return values.argmax(dim=-1).long()
```

This conversion is appropriate for `score`, but integer IDs are not
differentiable. The adapter cannot be used with gradient optimizers unless the
model is changed or wrapped with a float representation such as the weighted
embedding above.

For models requiring multiple inputs, wrap them in a module whose `forward`
accepts the single object returned by `prepare_input` and unpacks it internally.

`batch_size` controls discrete inference batches. `pad_to_length` and
`padding_side` can pad shorter strings with `N`; the equivalent soft input uses
all-zero vectors at padded positions.
