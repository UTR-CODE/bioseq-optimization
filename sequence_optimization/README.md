# Sequence Optimization API

See the repository-level [README](../README.md) for installation, Optimus,
DeepSTARR, fixed-budget, and q80/q90 saturation examples.

The package accepts a model through `CallableScorer`, `TorchSequenceScorer`, or
`DifferentiableTorchSequenceScorer`. A model may return one activity or a
matrix with one column per task.

All nucleotide sequences are normalized to uppercase `ACGT`; RNA `U` input is
converted to `T` internally.

Bundled checkpoints are loaded directly by name:

```python
from sequence_optimization import load_model

model = load_model("optimus5prime", device="cpu")
```

`available_models()` returns `optimus5prime`, `deepstarr`, `dream_legnet`,
`zhao_ptrc`, and `zhou_l3`. See the repository-level README for their sequence
lengths and outputs.

For a minimal user-defined PyTorch model, see
[`examples/custom_cnn.py`](examples/custom_cnn.py). It defines a one-layer CNN,
plain ACGT encoder, checkpoint loader, scorer wrapper, and genetic-algorithm
call without relying on BioSeqBench model code.

Available registry names:

- `random`
- `ism_greedy`
- `simulated_annealing`
- `genetic_algorithm`
- `adalead`
- `beam_search`
- `cross_entropy_method`
- `cbas`
- `vanilla_activation_maximization`
- `fast_seqprop`
- `den`

The first eight work with black-box activity scorers. Activation maximization,
Fast SeqProp, and DEN require a differentiable scorer that accepts soft
one-hot sequences. CbAS and DEN construct their generators internally.

For multi-task models, `TargetObjective(mode="maximize")` ignores non-target
outputs, while `TargetObjective(mode="specificity")` rewards the target and
penalizes other outputs.

Use `optimize_to_saturation(..., saturation=SaturationConfig(fractions=(0.8,
0.9)))` when a fixed query budget is not known in advance.
