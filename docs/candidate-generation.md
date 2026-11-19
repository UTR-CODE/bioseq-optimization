# Candidate generation

```python
from MotifSift import generate_designs

designs = generate_designs(model, length=200, objective="maximize", gain_fractions=(0.5, 0.9))
```

`designs` has one row per sequence, optimizer, and gain stage. Each row includes `sequence`, `algorithm`, `score`, and the model predictions in `activity_*` columns. When requesting multiple stages, use `gain_fraction` to select one. Identical sequences share a `sequence_id`.

Use `initial` to supply starting sequences. Otherwise, the optimizers start from a shared set of random sequences. Use `constraint` to supply a [sequence filter](examples/custom-constraints.md), or `fixed_region` to preserve positions from the starting sequence. With `is_CDS_optimized=True`, substitutions preserve the encoded protein.

See [objectives](core-api.md#choose-an-objective) for minimizing a score, weighting model outputs, or writing a custom score.
