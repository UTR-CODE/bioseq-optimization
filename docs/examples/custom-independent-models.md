# Multiple independent models

[Notebook](../../examples/custom-independent-models.ipynb)

Wrap each PyTorch model, then use `CombinedScorer` to score the same sequences with both. This example maximizes the difference between their outputs.

## Define the model

```python
import torch
from torch import nn
from MotifSift import CombinedScorer, SequencePredictor, generate_designs, random_seed_sequences


class ExampleModel(nn.Module):
    def __init__(self, patterns):
        super().__init__()
        self.conv = nn.Conv1d(4, len(patterns), kernel_size=3)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.fill_(-2)
            for channel, pattern in enumerate(patterns):
                for position, base in enumerate(pattern):
                    self.conv.weight[channel, "ACGT".index(base), position] = 1

    def forward(self, x):
        return self.conv(x).relu().sum(dim=-1)
```

## Connect the model and generate candidates

```python
target = SequencePredictor(ExampleModel(["ACG"]), tasks=["activity"])
off_target = SequencePredictor(ExampleModel(["GTA"]), tasks=["activity"])
model = CombinedScorer({"target": target, "off_target": off_target})


def objective(y):
    return y["target"] - y["off_target"]

length = 24
seeds = random_seed_sequences(length, 8)
designs = generate_designs(
    model, length=length, initial=seeds, objective=objective,
    num_candidates=8, score_threshold=2,
)
designs.groupby("algorithm")["score"].agg(["size", "median", "max"])
```
