# DREAM LegNet promoter activity

Load name: `dream_legnet`.

Maximize promoter_activity. Full construct length is 110 nt; variable-region design uses 93 nt.

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | cpu | Inference device |
| `checkpoint` | None | Optional checkpoint override |
| `weights_dir` | None | Optional model-weight root |
| `variable_region_only` | False | True for 93-nt variable region, False for 110-nt construct |
| `batch_size` | 4096 | Inference batch size |
| `pad_to_length` | None | Optional N-padding target; must satisfy model input requirements |
| `padding_side` | right | left or right |
| `task_names` | model output names | Optional output labels, without changing output order |

## Task

The recipe uses `variable_region_only=True` so the adapter supplies the fixed flanks. Full 110-nt design also changes those flanks unless the scorer constrains them.

```python
from sequence_optimization.examples.model_tasks import design_dream
result = design_dream(budget=100)
```

[Python task functions](../../sequence_optimization/examples/model_tasks.py).
Each recipe accepts `budget=100` and `device="cpu"`, uses four candidates and seed 7,
and returns an optimization result (Optimus returns an algorithm-to-result mapping).
Use larger budgets for design studies. A search direction is not an absolute activity threshold.
For mixed output scales, specify fixed reference centers/scales with [MultiTaskObjective](../objectives.md).
[Genetic optimizer parameters](../algorithms/genetic_algorithm.md).
