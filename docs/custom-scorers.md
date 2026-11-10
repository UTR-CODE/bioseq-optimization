# Custom sequence scorers

[Home](../README.md) · [Default PyTorch adapter](pytorch-models.md) · [Core API](core-api.md)

The built-in `SequencePredictor` assumes **four-channel A/C/G/T one-hot input** in that order, with shape `[batch,4,length]`. Passing a standard PyTorch module directly to `generate_designs` uses this adapter. Other channel orders, channel counts, tokenizations, feature encodings, and structured model inputs are connected through a custom scorer.

The optimization space remains DNA over A/C/G/T. A custom scorer maps that DNA to the predictor's own representation. The model can use a different number of input channels; the optimizer's soft input still represents the four possible DNA bases.

## 1. The two-method contract

| Method | Receives | Returns |
|---|---|---|
| `score(sequences)` | A list or NumPy array of DNA strings | NumPy activities `[batch,tasks]`; a single-task `[batch]` array also works |
| `score_soft(probabilities)` | A PyTorch tensor `[batch,length,4]` in A/C/G/T order | `(activities, None)`, with differentiable PyTorch activities `[batch,tasks]` |

`score` applies the predictor's own encoding and runs ordinary inference. `score_soft` converts continuous DNA probabilities into the model's representation using differentiable tensor operations. Its returned activities stay connected to the incoming probabilities, so the framework can calculate input gradients. Both methods use the same activity scale and task order.

Seven of the default optimizers use discrete scoring. DEN, Fast SeqProp, Ledidi, and activation maximization also use soft scoring. Implement both methods to use all eleven.

A scorer can optionally return prediction uncertainties: `score` accepts `(activities, uncertainties)` or `ScoreBatch`, and the second item from `score_soft` can be an uncertainty tensor. Set `task_names` for named optimization objectives. The `device` attribute identifies the scoring device; it is especially useful for GPU gradient batching.

## 2. Example: five channels in A/G/C/T/N order

Assume a trained model accepts `[batch,5,length]` in **A/G/C/T/N** order and returns one activity. The fifth channel represents N. Candidate designs use only A/C/G/T, so that N channel is zero for both discrete candidates and their soft representations.

```python
import numpy as np
import torch
import torch.nn.functional as F

class FiveChannelScorer:
    supports_soft_scoring = True

    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.task_names = ("activity",)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def score(self, sequences):
        # The predictor's native channel order is A/G/C/T/N.
        alphabet = {base: i for i, base in enumerate("AGCTN")}
        tokens = torch.tensor(
            [[alphabet[base] for base in str(seq).upper()] for seq in sequences],
            dtype=torch.long, device=self.device,
        )
        x = F.one_hot(tokens, num_classes=5).float().transpose(1, 2)
        with torch.no_grad():
            activities = self.model(x)
        return activities.detach().cpu().numpy().reshape(len(sequences), -1)

    def score_soft(self, probabilities):
        # Incoming channels always describe A/C/G/T DNA probabilities.
        p = probabilities.to(self.device)
        agct = p[:, :, [0, 2, 1, 3]]
        n_channel = torch.zeros_like(p[:, :, :1])
        x = torch.cat([agct, n_channel], dim=2).transpose(1, 2)
        activities = self.model(x)
        if activities.ndim == 1:
            activities = activities[:, None]
        return activities, None
```

The discrete method encodes strings directly in the predictor's native alphabet. The soft method reorders the optimizer's A/C/G/T probabilities and adds the model-specific channel while preserving gradients. For other feature encodings, replace these conversions with the model's corresponding discrete and differentiable transformations.

## 3. Use the scorer with the three core functions

```python
from sequence_optimization import generate_designs, call_motifs, select_candidates

device = "cuda" if torch.cuda.is_available() else "cpu"
scorer = FiveChannelScorer(model, device=device)

designs = generate_designs(
    scorer, length=200, objective=["maximize"],
    num_candidates=100, output_dir="outputs/custom_designs",
)
motifs = call_motifs(
    designs, model=scorer, output_dir="outputs/custom_motifs",
)
selected = select_candidates(
    designs, motifs=motifs, top_n=10, output_dir="outputs/custom_panel",
)
```

Here `model` is the trained five-channel predictor. The framework uses the supplied scorer methods directly. Scorers can be ordinary Python objects, as above, or PyTorch modules with their own scoring methods. They remain with the shared scoring service; optimizer processes send requests through the queue. In a standalone script, place model loading and workflow calls under `if __name__ == "__main__":`.

## 4. Check that both representations agree

At one-hot inputs, both methods should predict the same activities. A gradient should also reach the canonical DNA probabilities:

```python
from sequence_optimization import onehot

sequences = ["ACGT" * 50, "TGCA" * 50]
p = torch.tensor(onehot(sequences), dtype=torch.float32, device=device,
                 requires_grad=True)
soft, _ = scorer.score_soft(p)
np.testing.assert_allclose(
    soft.detach().cpu().numpy(), scorer.score(sequences),
    rtol=1e-4, atol=1e-5,
)
gradient, = torch.autograd.grad(soft.sum(), p)
print(gradient.shape)  # [2, 200, 4]
```

Apply identical normalization, fixed context, and output transformations in both paths. Keep `score_soft` in differentiable PyTorch operations; converting probabilities to integer tokens with `argmax`, detaching them, or converting them to NumPy removes the gradient path. For an embedding model, a soft path can form expected embeddings from nucleotide probabilities and the embedding matrix, then call a predictor entry point that accepts embeddings. Such a path is specified by the scorer rather than inferred by the framework.

## 5. Models with discrete prediction only

A predictor that only accepts strings or integer tokens can expose `score` and use the seven black-box optimizers:

```python
designs = generate_designs(
    scorer, length=200, objective=["maximize"], num_candidates=100,
    algorithms=[
        "adalead", "basesearch", "beam_search", "cross_entropy_method",
        "genetic_algorithm", "ism_greedy", "simulated_annealing",
    ],
)
```

To add the four gradient optimizers, provide a differentiable `score_soft` implementation or an explicit `differentiable_model` scorer. Sequence generation, motif discovery, and panel selection retain the same public functions.
