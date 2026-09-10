# UTR-CODE: 5′UTR, 3′UTR and CDS

Load name: `utr_code`.

Outputs log1p_rpf_rpkm_x5. Maximize predicted RPF at fixed RNA count and cell context. This is a translation-output proxy, not a stability prediction.

## Model loading parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | cpu | Inference device |
| `checkpoint` | None | Optional checkpoint override |
| `weights_dir` | None | Optional model-weight root |
| `design_region` | required | utr5, cds or utr3 |
| `utr5 / cds / utr3` | None | Provide the two non-designed regions |
| `cell_context` | HEK293T | Bundled context; other contexts require transcriptome_profile |
| `transcriptome_profile` | None | 13238 finite nonnegative raw expression values; adapter applies log1p(profile × 5) |
| `rna_count` | 6.3 | Fixed nonnegative RNA count |
| `batch_size` | 256 | Inference batch size |

## Task

The candidate replaces exactly one region; provide both other regions as fixed strings. Never supply a fixed string for design_region itself. Limits: utr5 1000 nt, cds 4500 nt, utr3 4000 nt.

```python
from sequence_optimization.examples.model_tasks import design_utr_code
utr5_result = design_utr_code("utr5")
utr3_result = design_utr_code("utr3")
cds_result = design_utr_code("cds", synonymous_codons={"A": ["GCT", "GCC"]})
```

The recipes use synthetic fixed flanks and a 60-nt reference CDS. Replace them with the transcript to design. [Synonymous CDS constraints](../tasks/synonymous_cds.md).

[Python task functions](../../sequence_optimization/examples/model_tasks.py).
Each recipe accepts `budget=100` and `device="cpu"`, uses four candidates and seed 7,
and returns an optimization result (Optimus returns an algorithm-to-result mapping).
Use larger budgets for design studies. A search direction is not an absolute activity threshold.
For mixed output scales, specify fixed reference centers/scales with [MultiTaskObjective](../objectives.md).
[Genetic optimizer parameters](../algorithms/genetic_algorithm.md).
