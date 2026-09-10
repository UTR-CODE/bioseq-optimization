# DeepSTARR Mouse tissue activity

Load name: `deepstarr_mouse_<tissue>_activity`.

1001 nt; independent activity models for six tissues.

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | cpu | Inference device |
| `checkpoint` | None | Optional checkpoint override |
| `weights_dir` | None | Optional model-weight root |
| `batch_size` | 4096 | Inference batch size |
| `pad_to_length` | None | Optional N-padding target; must satisfy model input requirements |
| `padding_side` | right | left or right |
| `task_names` | model output names | Optional output labels, without changing output order |

## Task

Tissues: heart, limb, midbrain, forebrain, hindbrain, neural_tube. The recipe maximizes the selected tissue and minimizes the other five.

```python
from sequence_optimization.examples.model_tasks import design_tissue
result = design_tissue("mouse", "heart")
```

Activity outputs are probabilities. Optional corresponding _accessibility models predict a different endpoint and are not substituted into this activity recipe.

[Python task functions](../../sequence_optimization/examples/model_tasks.py).
Each recipe accepts `budget=100` and `device="cpu"`, uses four candidates and seed 7,
and returns an optimization result (Optimus returns an algorithm-to-result mapping).
Use larger budgets for design studies. A search direction is not an absolute activity threshold.
For mixed output scales, specify fixed reference centers/scales with [MultiTaskObjective](../objectives.md).
[Genetic optimizer parameters](../algorithms/genetic_algorithm.md).
