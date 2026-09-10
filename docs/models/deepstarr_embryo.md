# DeepSTARR-Embryo tissue activity

Load name: `deepstarr_embryo_<tissue>_activity`.

1001 nt; independent tissue activity models.

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

The recipe combines brain_vnc, muscle, epidermis and gut, maximizing the target and minimizing the others.

```python
from sequence_optimization.examples.model_tasks import design_tissue
result = design_tissue("embryo", "brain_vnc")
```

The separate brain_specific_activity model predicts a specialized endpoint; it is not an additional tissue in this four-tissue comparison. Accessibility models exist for brain, cns, epidermis, midgut, somatic_muscle and visceral_muscle.

[Python task functions](../../sequence_optimization/examples/model_tasks.py).
Each recipe accepts `budget=100` and `device="cpu"`, uses four candidates and seed 7,
and returns an optimization result (Optimus returns an algorithm-to-result mapping).
Use larger budgets for design studies. A search direction is not an absolute activity threshold.
For mixed output scales, specify fixed reference centers/scales with [MultiTaskObjective](../objectives.md).
[Genetic optimizer parameters](../algorithms/genetic_algorithm.md).
