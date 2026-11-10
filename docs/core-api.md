# Three-function API

[Home](../README.md)

```python
from sequence_optimization import generate_designs, call_motifs, select_candidates
```

## 1. `generate_designs`

```python
designs = generate_designs(
    model, length=200, initial=None, objective=["maximize"],
    num_candidates=100, output_dir="outputs/designs", seed=42,
)
```

`model` may be a PyTorch `nn.Module`, `SequencePredictor`, or `CombinedScorer`. `initial=None` generates the shared random population. Select algorithms with `algorithms=[...]`; the default contains eleven non-random optimizers.

The default search runs to a plateau and retains the 90% observed-gain stage. `num_candidates` is the count per algorithm. The design table includes:

- `sequence`, `algorithm`, and scalar `score`.
- `activity_<task>` for each individual prediction output.
- `record_id` for each source record and `sequence_id` for each exact DNA sequence.

The output directory contains `designs.csv/fa`, `unique_designs.csv/fa`, `timing.csv`, and the run configuration.

```python
import pandas as pd
timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
```

## 2. `call_motifs`

```python
motifs = call_motifs(
    designs, model=model, output_dir="outputs/motifs",
    min_seqlets=10, min_parents=3,
)
```

Use the same model or scorer as in design generation. The design objective is carried in `designs.attrs` and reused for attribution. The default workflow is:

1. Combine exact-DNA duplicates.
2. Calculate ISM contributions for the design objective.
3. Call positive tangermeme seqlets.
4. Generate length- and dinucleotide-matched background fragments.
5. Discover candidate PWMs with STREME.
6. Retain motifs supported by at least ten seqlets from at least three parent sequences.

The returned `MotifCatalog` contains `pwms`, `seqlets`, and `support`. The output directory contains contributions, fragment sequences, the motif-support table, and `catalog.meme`.

To reuse attribution scores:

```python
motifs = call_motifs(designs, contributions=contributions)
```

A contribution array has shape `[input records, length]` in the same row order as `designs`. A contribution DataFrame can instead use `sequence_id` as its index. ISM and DeepLIFTShap projected contributions use the same interface.

To reuse a catalog:

```python
motifs = call_motifs(designs, catalog="catalog.meme")
```

Saved FIMO results can also be supplied with `hits=...`. After loading a design CSV, pass the original objective explicitly using `objective=...` when calculating new contributions.

## 3. `select_candidates`

### Default: Hamming distance

```python
selected = select_candidates(
    designs, top_n=10, min_score=None,
    output_dir="outputs/hamming_panel",
)
```

The selector combines exact-DNA duplicates, applies the score threshold, anchors on the highest score, and greedily maximizes the minimum Hamming distance to all previously selected designs. Near distance ties favor higher scores.

### With motifs: composition and positional diversity

```python
selected = select_candidates(
    designs, top_n=10, motifs=motifs, min_score=None,
    output_dir="outputs/motif_panel",
    min_sites=30, match_similarity=0.5,
    composition_tie=0.02, position_tie_bp=2.0,
)
```

`motifs` accepts the `MotifCatalog` returned by `call_motifs`, a MEME file, or a PWM dictionary. The selector performs FIMO scanning, physical-site consolidation, motif support filtering, PWM similarity, Soft-Jaccard composition distances, and Hungarian positional matching internally.

Both modes return the same type of selected design table:

```python
print(selected[["selection_rank", "sequence_id", "score", "algorithms"]])
print(selected.attrs["selection_method"])
print(selected.attrs["selection_trace"])
```

`min_score=None` includes the full finite-score pool. To restrict selection to high-quality designs, set an activity threshold or a score quantile before selection.
