# CDS stability

Load name: `cds_stability`.

Complete CDS, length divisible by three, at most 9000 nt. Maximize stability_score.

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | cpu | Inference device |
| `checkpoint` | None | Optional checkpoint override |
| `weights_dir` | None | Optional model-weight root |
| `species` | human | human, mouse, zebrafish, fruitfly, yeast |
| `codonbert_model_dir` | None | Local CodonBERT config/weights directory |
| `batch_size` | 4 | Inference batch size |

## Task

Preserve the encoded protein with reference_cds. The CodonBERT branch reads up to 3066 nt; longer inputs are permitted by the full model but not fully represented in that branch.

```python
from sequence_optimization.examples.model_tasks import design_cds_stability
stable_cds = design_cds_stability(synonymous_codons={"A": ["GCC", "GCG"]})
joint = design_cds_stability(joint=True)
```

The joint task sends each CDS to CDS Stability and UTR-CODE with fixed 5′/3′UTRs, maximizing [stability_score, RPF]. It targets stability and translation output, not two independent stability estimates. Joint candidates must satisfy the smaller 4500-nt UTR-CODE limit.

[Python task functions](../../sequence_optimization/examples/model_tasks.py).
Each recipe accepts `budget=100` and `device="cpu"`, uses four candidates and seed 7,
and returns an optimization result (Optimus returns an algorithm-to-result mapping).
Use larger budgets for design studies. A search direction is not an absolute activity threshold.
For mixed output scales, specify fixed reference centers/scales with [MultiTaskObjective](../objectives.md).
[Genetic optimizer parameters](../algorithms/genetic_algorithm.md).
