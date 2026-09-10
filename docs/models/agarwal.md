# Agarwal lentiMPRA cell specificity

Load name: `agarwal_k562 / agarwal_hepg2 / agarwal_wtc11`.

230 nt. Each independent model predicts log2_rna_dna.

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

The combined order is K562, HepG2, WTC11. K562 specificity uses ["maximize", "minimize", "minimize"].

```python
from sequence_optimization.examples.model_tasks import design_agarwal
result = design_agarwal("k562")  # also "hepg2" or "wtc11"
```

Default prediction averages ten folds and both orientations; providing checkpoint selects one checkpoint. Gradient conversion requires A/G/C/T channel order.

[Python task functions](../../sequence_optimization/examples/model_tasks.py).
Each recipe accepts `budget=100` and `device="cpu"`, uses four candidates and seed 7,
and returns an optimization result (Optimus returns an algorithm-to-result mapping).
Use larger budgets for design studies. A search direction is not an absolute activity threshold.
For mixed output scales, specify fixed reference centers/scales with [MultiTaskObjective](../objectives.md).
[Genetic optimizer parameters](../algorithms/genetic_algorithm.md).
