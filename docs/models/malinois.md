# Malinois cell-line-specific enhancers

Load name: `malinois`.

200-nt variable enhancer. Output order: K562_log2FC, HepG2_log2FC, SKNSH_log2FC.

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

The adapter inserts fixed left/right reporter context, forming a 600-nt model input. Optimize only the 200-nt candidate.

```python
from sequence_optimization.examples.model_tasks import design_malinois
result = design_malinois("K562")  # also "HepG2" or "SKNSH"
```

K562 specificity uses ["maximize", "minimize", "minimize"]. Gradient optimization needs a differentiable context-insertion adapter; the recipe uses genetic search.

[Python task functions](../../sequence_optimization/examples/model_tasks.py).
Each recipe accepts `budget=100` and `device="cpu"`, uses four candidates and seed 7,
and returns an optimization result (Optimus returns an algorithm-to-result mapping).
Use larger budgets for design studies. A search direction is not an absolute activity threshold.
For mixed output scales, specify fixed reference centers/scales with [MultiTaskObjective](../objectives.md).
[Genetic optimizer parameters](../algorithms/genetic_algorithm.md).
