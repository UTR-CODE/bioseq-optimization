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

`model` may be a PyTorch `nn.Module`, `SequencePredictor`, or `CombinedScorer`. A plain module is automatically wrapped: its `forward` accepts `[batch,4,length]` A/C/G/T inputs and returns `[batch]` or `[batch,tasks]` activity tensors. The wrapper supplies both discrete `score` and differentiable `score_soft`; users do not implement those methods on a standard CNN. `initial=None` generates the shared random population. Select algorithms with `algorithms=[...]`; the default contains eleven non-random optimizers.

By default, `parallel=True` starts one CPU optimization process per algorithm. A single scoring service in the calling process retains the loaded model, receives requests through a shared queue, and combines compatible discrete or differentiable requests into batches. Model objects stay out of the worker processes.

The default search runs to a plateau and retains the 90% observed-gain stage. `num_candidates` is the count per algorithm. The design table includes:

- `sequence`, `algorithm`, and scalar `score`.
- `activity_<task>` for each individual prediction output.
- `record_id` for each source record and `sequence_id` for each exact DNA sequence.

The output directory contains `designs.csv/fa`, `unique_designs.csv/fa`, `timing.csv`, and the run configuration. Parallel runs also export `gpu_batches.csv/json`; the returned table includes `designs.attrs["batching"]` and `designs.attrs["execution"]`.

```python
import pandas as pd
timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
```

Batch sizes and the collection window can be adjusted through the same function:

```python
designs = generate_designs(
    model, length=200, num_candidates=100,
    max_batch_size=512, max_soft_batch_size=64, batch_wait_ms=5.0,
    output_dir="outputs/batched_designs",
)
print(designs.attrs["batching"])
```

`parallel=False` uses sequential optimization for comparison. Per-algorithm parallel runtimes include queue waiting; `wall_seconds` measures the whole generation call. Notebooks can call the function directly. In standalone Python scripts, place model loading and workflow calls under `if __name__ == "__main__":`.

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
5. Discover candidate PWMs with official STREME.
6. Scan the seqlets with FIMO and retain motifs supported by at least ten seqlets from at least three parent sequences.

The default `motif_backend="streme"` uses official STREME and FIMO. Missing executables produce an installation message before attribution starts. The optional `motif_backend="builtin"` uses the independently implemented discovery method described in [Motif discovery](motif-discovery.md).

The returned `MotifCatalog` contains `pwms`, `seqlets`, and `support`. With `output_dir="outputs/motifs"`, the function exports the retained PWMs to **`outputs/motifs/catalog.meme`** in MEME format, alongside contributions, fragment sequences, and the motif-support table. This file can be passed directly to `select_candidates` in a later session.

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

### Use your own motif-discovery workflow

The discovery and selection stages communicate through a MEME file. Discover motifs with your preferred method, export DNA letter-probability matrices, and pass the path directly to selection. The file must declare `ALPHABET= ACGT` and use unique motif IDs.

```python
import pandas as pd

designs = pd.read_csv("outputs/designs/designs.csv")
gate = float(designs.drop_duplicates("sequence").score.quantile(0.9))
selected = select_candidates(
    designs,
    motifs="my_motifs.meme",  # Also accepts pathlib.Path
    top_n=10,
    min_score=gate,
    output_dir="outputs/custom_motif_panel",
)
```

This route needs the scored candidate sequences and FIMO. It does not require `call_motifs`, the prediction model, attribution scores, tangermeme, or STREME. FIMO identifies motif IDs and positions internally, followed by the same site-support filtering and diversity selection. The MEME file exported by step 2 works through exactly this interface:

```python
selected = select_candidates(
    designs, motifs="outputs/motifs/catalog.meme", top_n=10,
    min_score=gate, output_dir="outputs/exported_motif_panel",
)
```

Both modes return the same type of selected design table:

```python
print(selected[["selection_rank", "sequence_id", "score", "algorithms"]])
print(selected.attrs["selection_method"])
print(selected.attrs["selection_trace"])
```

`min_score=None` includes the full finite-score pool. To restrict selection to high-quality designs, set an activity threshold or a score quantile before selection.
