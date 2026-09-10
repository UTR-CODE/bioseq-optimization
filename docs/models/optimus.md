# Optimus 5-Prime

Load name: `optimus5prime`.

50 nt; output mrl. Maximize mean ribosome load.

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

Use `optimus_algorithms(["genetic_algorithm", "basesearch", "simulated_annealing"])` to compare optimizers with separate budgets. Gradient methods use the model's channels-first differentiable input. For ISM, allow at least 154 queries for four seeds plus a complete neighborhood.

```python
from sequence_optimization.examples.model_tasks import optimus_algorithms
results = optimus_algorithms(["genetic_algorithm", "basesearch"], budget=200)
for name, result in results.items():
    print(name, result.to_frame())
```

[Python task functions](../../sequence_optimization/examples/model_tasks.py).
Each recipe accepts `budget=100` and `device="cpu"`, uses four candidates and seed 7,
and returns an optimization result (Optimus returns an algorithm-to-result mapping).
Use larger budgets for design studies. A search direction is not an absolute activity threshold.
For mixed output scales, specify fixed reference centers/scales with [MultiTaskObjective](../objectives.md).
[Genetic optimizer parameters](../algorithms/genetic_algorithm.md).
