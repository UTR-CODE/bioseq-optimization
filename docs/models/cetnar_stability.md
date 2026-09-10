# Cetnar mRNA stability

Load name: `cetnar_stability`.

Up to 80 nt. Output decay_rate_k_fit is a decay rate: minimize it for increased predicted stability.

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | cpu | Inference device |
| `checkpoint` | None | Optional checkpoint override |
| `weights_dir` | None | Optional model-weight root |
| `batch_size` | 4096 | Inference batch size |
| `pad_to_length` | 80 | Optional N-padding target; must satisfy model input requirements |
| `padding_side` | right | left or right |
| `task_names` | model output names | Optional output labels, without changing output order |

## Task

The model is a benchmark surrogate for short-sequence decay, not a complete-transcript stability model.

```python
from sequence_optimization.examples.model_tasks import design_cetnar
stable = design_cetnar()
joint = design_cetnar(joint=True)
```

In the joint recipe, the same 80-nt candidate is scored by Cetnar and inserted as UTR-CODE's 3′UTR, with fixed 5′UTR/CDS. Directions are ["minimize", "maximize"] for [decay_rate, RPF]. This combines distinct assay contexts; predictions alone do not establish full-transcript stability or translation efficiency. Use CPU for gradient inference in eval mode.

[Python task functions](../../sequence_optimization/examples/model_tasks.py).
Each recipe accepts `budget=100` and `device="cpu"`, uses four candidates and seed 7,
and returns an optimization result (Optimus returns an algorithm-to-result mapping).
Use larger budgets for design studies. A search direction is not an absolute activity threshold.
For mixed output scales, specify fixed reference centers/scales with [MultiTaskObjective](../objectives.md).
[Genetic optimizer parameters](../algorithms/genetic_algorithm.md).
