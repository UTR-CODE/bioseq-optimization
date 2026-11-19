# Candidate generation

```python
from MotifSift import generate_designs

designs = generate_designs(model, length=200, objective="maximize")
```

`designs` has one row per sequence, optimizer, and gain stage. Each row includes `sequence`, `algorithm`, `score`, and the model predictions in `activity_*` columns. When requesting multiple stages, use `gain_fraction` to select one. See [Choosing a gain fraction](gain-fraction.md) for the calculation and when each candidate set is selected. Set `score_threshold` to run toward a target score. The two search modes are mutually exclusive. Identical sequences share a `sequence_id`.

Use `initial` to supply starting sequences. Otherwise, the optimizers start from a shared set of random sequences. Use `constraint` to supply a [sequence filter](examples/custom-constraints.md), or `fixed_region` to preserve positions that agree across all starting sequences. See [Keep selected positions fixed](examples/custom-constraints.md#keep-selected-positions-fixed) for a working example. With `is_CDS_optimized=True`, substitutions preserve the encoded protein.

See [objectives](core-api.md#choose-an-objective) for minimizing a score, weighting model outputs, or writing a custom score.

The default optimization seed is 42. Each optimizer uses a deterministic offset from that seed in the order listed below. Pass `seed=` to change the base seed. When comparing runs, match the starting sequences, model checkpoint, prediction precision, and gain stage.

## Optimizers

MotifSift runs eleven optimizers over a common starting pool: AdaLead, BaseSearch, beam search, cross-entropy, DEN, Fast SeqProp, genetic algorithm, in silico mutagenesis, Ledidi, simulated annealing, and vanilla activation maximization.

Each contributes up to `num_candidates` sequences per requested gain stage. To run a subset, pass the optimizer names in `algorithms`.
