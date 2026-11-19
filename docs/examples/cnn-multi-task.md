# Multi-output CNN

[Notebook](../../examples/cnn-multi-task.ipynb)

Wrap one PyTorch CNN that returns several scores per sequence. The two filters count `ACG` and `GTA` in output order. Increase `target` and decrease `off_target` with twice the weight on `target`.

## Define the model

```python
import torch
from torch import nn
from MotifSift import MultiTaskObjective, SequencePredictor, generate_designs, random_seed_sequences


class TinyCNN(nn.Module):
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
model = SequencePredictor(TinyCNN(["ACG", "GTA"]), tasks=["target", "off_target"])
objective = MultiTaskObjective(
    high=["target"], low=["off_target"], weights={"target": 2},
)

length = 24
seeds = random_seed_sequences(length, 8)
designs = generate_designs(
    model, length=length, initial=seeds, objective=objective,
    num_candidates=8, score_threshold=1,
)
designs.groupby("algorithm")["score"].agg(["size", "median", "max"])
```
