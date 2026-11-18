# Candidate generation

[Home](../README.md) · [Core API](core-api.md)

`generate_designs` is the generation stage of the three-function framework.

## Shared starting sequences

```python
from sequence_optimization import generate_designs, random_seed_sequences
initial = random_seed_sequences(200, 100, seed=42)
designs = generate_designs(
    model, length=200, initial=initial, objective=["maximize"],
    num_candidates=100, seed=42, output_dir="outputs/designs",
)
```

The starting population is shared across all selected algorithms. Omitting `initial` generates it automatically. Each algorithm contributes 100 designs; the default eleven-algorithm pool contains 1,100 source records.

Parallel generation refreshes the progress display every five seconds. Each optimizer shows its current observation round and the mean objective score of its retained candidate archive. Finished optimizers are marked as completed, and redirected output receives periodic snapshots. Pass `show_progress=False` to suppress the display.

## Search modes

Choose one stopping mode for each run.

- **Observed gain** is the default. MotifSift runs to its plateau and returns the first complete candidate archive whose mean reaches `initial_mean + gain_fraction × (plateau_mean − initial_mean)`. Set `gain_fraction=0.5` to select the 50% gain stage. The default is `0.9`.
- **Absolute score threshold** uses `score_threshold=...`. MotifSift stops when the best feasible score in a complete candidate archive reaches that value. Other candidates in the returned archive can be below the threshold. If the search terminates before reaching it, the run reports the termination reason and `threshold_reached=False`.

```python
designs = generate_designs(model, 200, objective=["maximize"], gain_fraction=0.5)
designs = generate_designs(model, 200, objective=["maximize"], gain_fractions=(0.5, 0.9))
designs = generate_designs(model, 200, objective=["maximize"], score_threshold=1.2)
```

With multiple gain fractions, each optimizer searches once per starting reference and returns up to `num_candidates` candidates for each requested stage. Use the `gain_fraction` column to select a stage in the returned table or `designs.csv`. The `unique_designs.csv` export deduplicates across all stages. `timing.csv` counts the shared search once per optimizer, and `gain_stage_timing.csv` records the search total and the earlier selection query count for each stage and reference. Each target uses the terminal plateau mean. The knee is recorded as a diagnostic.

The command-line search accepts repeated flags, for example `--gain-fraction 0.5 --gain-fraction 0.9`.

Thresholds are on the scalar optimization objective. Use an increasing scalar objective to represent improvement. For example, to reach a raw value of at most 0.2 with `objective=["minimize"]`, pass `score_threshold=-0.2`. Per-task `stop_values` remain feasibility bounds. Query counts are diagnostics, and fixed query budgets are no longer accepted.

## Select optimization methods

```python
designs = generate_designs(
    model, 200, objective=["maximize"], num_candidates=100,
    algorithms=["genetic_algorithm", "fast_seqprop"],
)
```

This example returns 200 candidate records. [Algorithm names](algorithms.md) can be mixed in the same call.

## Runtime and outputs

```python
import pandas as pd
timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
```

Query counts include repeated optimization proposals; shared prediction caching reuses model evaluations. The design table carries the individual model outputs, scalar score, algorithm identity, record ID, and stable sequence ID. An output directory stores the full and unique sequence tables and FASTA files, runtime summary, and run configuration.

Continue with `call_motifs(designs, model=...)`, then `select_candidates(designs, motifs=...)`. To select using sequence distance alone, call `select_candidates(designs)` directly.

## Shared batched model scoring

`generate_designs` uses parallel optimization by default. Each algorithm has a CPU process and sends discrete sequence or differentiable scoring requests to a shared queue. One service in the calling process retains the model, batches compatible requests, and routes scores or input gradients back to the requesting worker. The shared discrete cache reuses predictions across algorithms.

```python
print(designs.attrs["execution"])  # parallel
print(designs.attrs["batching"])
```

Use `max_batch_size`, `max_soft_batch_size`, and `batch_wait_ms` to control batching; see [Parallel scoring](parallel-scoring.md). In a standalone script, place model loading and workflow calls under `if __name__ == "__main__":`. Set `parallel=False` to compare against sequential optimization.
