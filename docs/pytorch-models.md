# PyTorch model interface

[Home](../README.md) · [Candidate generation](candidate-generation.md)

The optimizer operates on nucleotide sequences. `SequencePredictor` connects a PyTorch model to discrete scoring and differentiable sequence optimization.

| Component | Interface |
|---|---|
| Model | Any `torch.nn.Module` implementing sequence-to-activity prediction |
| Default input | Float32 `[batch, 4, length]`, A/C/G/T channels |
| Model output | `[batch]` or `[batch, tasks]` activity tensor |
| Predictor output | `[batch, tasks]` activity matrix |
| Canonical probabilities | `[batch, length, 4]` |
| Input adaptation | `input_transform(probabilities)` |
| Output adaptation | `output_transform(model_output)` |

## 1. A sequence-to-activity network

This compact architecture illustrates the interface. Supply the trained model from your prediction workflow when designing experimental sequences.

```python
import torch
from torch import nn
from sequence_optimization import SequencePredictor

class ActivityCNN(nn.Module):
    def __init__(self, outputs=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, outputs),
        )

    def forward(self, x):
        return self.network(x)

model = ActivityCNN(outputs=1)
scorer = SequencePredictor(model, tasks=["activity"], device="cpu")
```

Model loading remains part of the prediction workflow. For example, load a trained architecture and its state dictionary, or reuse an already loaded `nn.Module`:

```python
# model = YourActivityNetwork(...)
# model.load_state_dict(torch.load("checkpoint.pt", weights_only=True))
scorer = SequencePredictor(model, tasks=["activity"], device="cpu")
```

## 2. Input layouts

A model accepting `[batch, length, 4]` uses an identity input transform:

```python
scorer = SequencePredictor(
    model, tasks=["activity"], device="cpu",
    input_transform=lambda p: p,
)
```

For another channel order, reorder the canonical A/C/G/T probabilities before converting to channels-first:

```python
scorer = SequencePredictor(
    model, tasks=["activity"], device="cpu",
    input_transform=lambda p: p[:, :, [0, 2, 1, 3]].transpose(1, 2),
)
```

Input transforms can also apply padding, concatenate fixed contexts, or map probabilities into model embeddings. They receive probabilities on the selected model device. Use PyTorch tensor operations to carry input gradients through the transformation.

## 3. Structured outputs

For a model returning a dictionary, select its activity tensor:

```python
scorer = SequencePredictor(
    model, tasks=["activity"], device="cpu",
    output_transform=lambda result: result["activity"],
)
```

For multiple outputs, name the tasks and define the optimization objective separately:

```python
from sequence_optimization import MultiTaskObjective
scorer = SequencePredictor(model, tasks=["target", "off_target"], device="cpu")
objective = MultiTaskObjective(directions=["maximize", "minimize"])
```

An output transform can construct nonlinear contrasts such as target activity minus the largest off-target activity; see the [enhancer example](examples/enhancer.md).

## 4. Sequence scoring and gradients

```python
from sequence_optimization import random_seed_sequences, onehot
initial = random_seed_sequences(200, 4, seed=42)
activities = scorer.score(initial).scores

p = torch.tensor(onehot(initial), requires_grad=True)
soft_activities, _ = scorer.score_soft(p)
gradient, = torch.autograd.grad(soft_activities.sum(), p)
```

Both paths use the same model and transformations. The model is evaluated with fixed weights; gradient algorithms update sequence variables.

## 5. String-based predictors

Black-box algorithms can also use a function mapping sequence strings to activity values:

```python
from sequence_optimization import CallableScorer
scorer = CallableScorer(predict_sequences, task_names=["activity"])
```

Here `predict_sequences` is your own prediction function. Select black-box algorithms for this interface, or provide a corresponding differentiable model for gradient algorithms.
