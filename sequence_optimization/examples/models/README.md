# Bundled checkpoint models

The public `load_model()` registry provides:

- `optimus5prime`: single-task 5′UTR MRL prediction;
- `deepstarr`: two-task developmental/housekeeping enhancer activity;
- `dream_legnet`: DREAM 2022 yeast promoter activity;
- `zhao_ptrc`: Ptrc promoter log10 fluorescence/OD600;
- `zhou_l3`: fixed-length L3 promoter activity.

Each adapter provides `from_checkpoint()` and loads its included checkpoint by
default. Most users should use the shorter public interface:

```python
from sequence_optimization import load_model

model = load_model("deepstarr", device="cpu")
```

An explicit checkpoint path can be supplied when using another trained copy.

The optimizer itself is model-independent. A different model can implement
`score(sequences)` or be wrapped with `TorchSequenceScorer`; differentiable
models can use `DifferentiableTorchSequenceScorer`.
