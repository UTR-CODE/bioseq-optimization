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

## Plateau and gain-stage selection

```python
from sequence_optimization import SaturationConfig
saturation = SaturationConfig(
    fractions=(0.9,), selection="fraction", maximum_queries=None,
)
```

Pass `saturation` to `generate_designs` to configure stopping explicitly. The standard DNA workflow uses this configuration by default. The gain stage is measured relative to the initial and final archive quality observed in each run.

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
