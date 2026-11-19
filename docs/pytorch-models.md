# Connect your own PyTorch model

Models from `bioseq_models` already implement MotifSift's scoring interface and can be used directly.

For your own PyTorch model, wrap it once with `SequencePredictor(model, tasks)`. List task names in the same order as the model outputs. The adapter implements `score()` for nucleotide strings and `score_soft()` for differentiable sequence optimization. It uses the device on which you placed the model.

```python
from MotifSift import SequencePredictor, generate_designs

scorer = SequencePredictor(my_model, tasks=["activity"])
designs = generate_designs(scorer, length=200, objective="maximize")
```

By default, `forward()` receives A/C/G/T one-hot input of shape `[batch, 4, length]` and returns a tensor of shape `[batch]` or `[batch, tasks]`.

## Model with a different input layout

For a model trained on `[batch, length, 4]` in A/C/G/T order, override `prepare_input`. It receives base probabilities in that layout, so this example passes them through. Use PyTorch operations for layout changes so the gradients reach the sequence probabilities.

```python
from MotifSift import SequencePredictor


class ChannelsLastPredictor(SequencePredictor):
    def prepare_input(self, probabilities):
        return probabilities


scorer = ChannelsLastPredictor(my_model, tasks=["activity"])
```

If your model returns a dictionary or other structured output, make its PyTorch `forward()` return the activity tensor. To combine multiple activity outputs into an optimization score, use a [custom objective](core-api.md#choose-an-objective).

Use the model-specific input examples for a [strand-specific 5′ UTR model](templates/5utr.md) or a [promoter model with fixed flanks](templates/promoter.md).
