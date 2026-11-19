# Model with custom scoring methods

[Notebook](../../examples/custom-scoring-methods.ipynb)

Pass an object that implements `score()` and `score_soft()` directly to MotifSift. These two methods are enough for a single-output model. This example scores the number of G bases. Apply the same prediction logic to strings and probabilities in your own model.

## Define the scoring methods

```python
from MotifSift import generate_designs, random_seed_sequences


class GCountScorer:
    def score(self, sequences: list[str]):
        return [sequence.count("G") for sequence in sequences]

    def score_soft(self, probabilities):
        return probabilities[:, :, 2].sum(dim=1)
```

`score()` receives normalized strings in a Python list. MotifSift uppercases the input and converts U to T. N represents an unknown base. For `score_soft()`, the input is a PyTorch tensor of shape `[batch, length, 4]` in A/C/G/T order. Channel 2 contains G probabilities. The returned tensor stays connected to the inputs for gradient-based optimization.

## Generate candidates

```python
length = 24
seeds = random_seed_sequences(length, 8)
designs = generate_designs(
    GCountScorer(), length=length, initial=seeds, objective="maximize",
    num_candidates=8, score_threshold=14,
)
designs.groupby("algorithm")["score"].agg(["size", "median", "max"])
```
