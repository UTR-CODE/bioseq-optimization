# DeepSTARR: housekeeping high, developmental low

Load name: `deepstarr`.

249 nt. Output order: developmental, housekeeping.

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

Use `objective=["minimize", "maximize"]`.

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize
model = load_model("deepstarr", device="cpu")
result = genetic_algorithm_optimize(model, length=249,
    objective=["minimize", "maximize"], num_candidates=4, budget=100, seed=7)
```

These are two reporter tasks, not two cell lines.

[Python task functions](../../sequence_optimization/examples/model_tasks.py).
Each recipe accepts `budget=100` and `device="cpu"`, uses four candidates and seed 7,
and returns an optimization result (Optimus returns an algorithm-to-result mapping).
Use larger budgets for design studies. A search direction is not an absolute activity threshold.
For mixed output scales, specify fixed reference centers/scales with [MultiTaskObjective](../objectives.md).
[Genetic optimizer parameters](../algorithms/genetic_algorithm.md).
