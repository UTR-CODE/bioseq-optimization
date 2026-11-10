# Candidate generation

[Home](../README.md) · [Enhancer](examples/enhancer.md) · [Promoter](examples/promoter.md) · [5′UTR](examples/5utr.md)

Each optimizer starts from the same population of 100 sequences and produces 100 retained candidates. Eleven optimizers yield 1,100 candidate records. Exact duplicates are then combined into a unique sequence pool with algorithm provenance.

## 1. Shared starting population

```python
from sequence_optimization import random_seed_sequences, SaturationConfig
from sequence_optimization.workflows import DESIGN_ALGORITHMS

length = 200
initial = random_seed_sequences(length, 100, seed=42)
saturation = SaturationConfig(
    fractions=(0.9,), selection="fraction", maximum_queries=None,
)
```

The search stops when the candidate archive reaches a plateau. The 90% gain stage is defined relative to the initial and final archive quality observed within each run.

## 2. Optimization loop

Use the `scorer` and `objective` created in the task examples.

```python
import time
import pandas as pd
from sequence_optimization import optimize

frames, timing = [], []
for algorithm in DESIGN_ALGORITHMS:
    start = time.perf_counter()
    result = optimize(
        scorer, length, initial, objective, 100,
        algorithm=algorithm, seed=42, diversity=5,
        saturation=saturation, count_repeated=True,
    )
    elapsed = time.perf_counter() - start
    retained = result.at(0.9)
    frame = pd.DataFrame({
        "record_id": [f"{algorithm}_{i+1:03}" for i in range(len(retained.sequences))],
        "sequence": retained.sequences,
        "algorithm": algorithm,
        "score": retained.scores,
    })
    for j, task in enumerate(scorer.task_names):
        frame[f"activity_{task}"] = retained.activities[:, j]
    frames.append(frame)
    timing.append({"algorithm": algorithm, "seconds": elapsed,
                   "queries": result.total_query_count, "stop_reason": result.stop_reason})

candidates = pd.concat(frames, ignore_index=True)
timing = pd.DataFrame(timing)
```

`diversity=5` sets the minimum Hamming distance within each optimizer's retained panel.

## 3. Generate and save the complete candidate pool

The reusable workflow runs the same loop and saves the seed population, configuration, trajectories, candidates, and timing table.

```python
from sequence_optimization.workflows import generate_candidates

candidates, timing = generate_candidates(
    scorer, length, initial, objective,
    output_dir="outputs/design", num_candidates=100,
    saturation=saturation, diversity=5, seed=42,
)
print(timing[["algorithm", "seconds", "queries", "stop_reason"]])
```

## 4. Unique sequences and provenance

```python
from sequence_optimization.workflows import deduplicate_candidates, write_fasta
unique, provenance = deduplicate_candidates(candidates)
write_fasta("outputs/design/pool.fa", unique.sequence_id, unique.sequence)
```

`record_id` identifies a candidate within an algorithm. `sequence_id` identifies exact DNA by a sequence hash. `algorithms` lists all algorithms producing each unique sequence.

| Output | Contents |
|---|---|
| `all_candidates.csv` | All 1,100 candidate records and model outputs |
| `unique_candidates.csv/fa` | Unique DNA with algorithm provenance |
| `algorithm_summary.csv` | Runtime, query count, stopping reason, score summaries |
| `<algorithm>/trajectory.json` | Archive quality throughout optimization |

## 5. Random-search reference

Generate and score 200,000 random sequences separately, using the same scalar objective as the optimization runs.

```python
import numpy as np
pool = random_seed_sequences(length, 200_000, seed=43)
activities = scorer.score(pool).scores
scores = (activities[:, 0] if activities.shape[1] == 1
          else objective.scalarize(activities, scorer.task_names))
top = np.argsort(-scores, kind="stable")[:100]
random_reference = pd.DataFrame({"sequence": pool[top], "score": scores[top]})
```

For a single-output model, set `scores = activities[:, 0]`.
