# Parallel optimization and shared model batching

[Home](../README.md) · [Core API](core-api.md) · [Candidate generation](candidate-generation.md)

`generate_designs` runs independent optimization algorithms in separate CPU processes by default. The function name, model argument, and returned design table are the same for sequential and parallel execution.

```text
CPU optimizer 1 ─┐                       ┌─ activity predictions
CPU optimizer 2 ─┼→ shared request queue → batch → model on GPU/CPU
CPU optimizer N ─┘                       └─ gradients for sequence updates
```

## 1. Use the same model and function

With a loaded PyTorch model or sequence scorer:

```python
from sequence_optimization import generate_designs

designs = generate_designs(
    model, length=200, objective=["maximize"],
    num_candidates=100, seed=42,
    max_batch_size=512, max_soft_batch_size=64, batch_wait_ms=5.0,
    output_dir="outputs/parallel_designs",
)
```

The 11 optimizers share an initial random population, run to a plateau, and retain 100 candidates each at 90% observed gain. The loaded model stays in the calling process. CPU workers receive queue clients and sequence data, so notebook-defined model classes and already-loaded CUDA models can use the same interface. A service thread owns model evaluation; workers do not receive model objects.

The service merges compatible requests of three types: discrete DNA prediction, differentiable forward prediction, and differentiable backward evaluation. Backward requests recompute the model output and return input gradients to the optimizer. The shared discrete cache avoids scoring the same DNA repeatedly across algorithms. Independent black-box and differentiable scorers can be supplied using the existing `differentiable_model` argument.

## 2. Tune batch sizes

| Parameter | Default | Role |
|---|---|---|
| `parallel` | `True` | Separate CPU optimizer processes and shared scoring |
| `max_batch_size` | `512` | Maximum discrete scoring batch |
| `max_soft_batch_size` | `64` | Maximum differentiable scoring batch |
| `batch_wait_ms` | `5.0` | Time window for collecting compatible requests |

Workers use one PyTorch, Numba, and BLAS thread each. Models retain their configured device; the same scheduling works with CPU inference. Set `parallel=False` to run the sequential reference path.

## 3. Inspect runtime and utilization

```python
import pandas as pd

timing = pd.DataFrame(designs.attrs["timing"])
print(timing[["algorithm", "seconds", "cpu_seconds", "queries", "stop_reason"]])
print("Total generation seconds:", designs.attrs["wall_seconds"])
print("Batching:", designs.attrs["batching"])
```

`timing.csv` includes the optimizer process ID. Algorithm seconds include scoring and queue waiting. `gpu_batches.csv` records request kind, rows, number of workers combined, queue waiting, and scoring time for every batch. `gpu_batches.json` summarizes total rows, batch sizes, cross-worker batches, and scoring time. These metrics are also returned in `designs.attrs["batching"]`.

## 4. Run notebooks and scripts

Notebook cells can call `generate_designs` directly. In standalone Python scripts, put model loading and the workflow in the main entry point. For example, load a trained TorchScript predictor:

```python
import torch
from sequence_optimization import generate_designs

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load("activity_model.ts", map_location=device).eval()
    designs = generate_designs(
        model, length=200, objective=["maximize"],
        num_candidates=100, output_dir="outputs/script_designs",
    )
```

Workers use `spawn`. Custom objective or constraint functions supplied to workers are defined at module scope in an importable module. Model classes and input/output transforms stay with the scoring service and do not need to be serialized to workers.

## 5. Continue motif discovery and candidate selection

The returned table carries the optimization objective, predictions, and provenance into the next two stages:

```python
from sequence_optimization import call_motifs, select_candidates

motifs = call_motifs(designs, model=model, output_dir="outputs/parallel_motifs")
selected = select_candidates(
    designs, motifs=motifs, top_n=10, output_dir="outputs/parallel_panel",
)
```

A MEME file can be passed directly to `select_candidates` as in the sequential workflow.
