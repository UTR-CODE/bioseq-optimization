# Parallel optimization

[Home](../README.md) · [Candidate generation](candidate-generation.md)

The parallel backend accelerates the first stage of the three-function framework. Independent CPU workers generate sequence proposals. A shared scoring process collects their requests, combines compatible requests into batches, and evaluates the PyTorch model on the GPU. Gradient requests use the same service.

## 1. Define a scoring factory

The factory creates the model and `SequencePredictor` inside the scoring process. Use your experiment's model-loading function. Place the factory in an importable module, for example `experiment_model.py`:

```python
from sequence_optimization import SequencePredictor
from my_prediction_workflow import load_trained_model

def make_scorer():
    model = load_trained_model()
    return SequencePredictor(model, tasks=["activity"], device="cuda", batch_size=512)
```

## 2. Run the optimization workers

The following driver launches 11 CPU optimization workers and one scoring process:

```python
from sequence_optimization import random_seed_sequences
from sequence_optimization.workflows import generate_candidates_parallel
from experiment_model import make_scorer

if __name__ == "__main__":
    initial = random_seed_sequences(200, 100, seed=42)
    designs, timing = generate_candidates_parallel(
        make_scorer, 200, initial, ["maximize"],
        task_names=["activity"], output_dir="outputs/parallel_design",
        num_candidates=100, max_batch_size=512,
        max_soft_batch_size=64, batch_wait_ms=5.,
    )
    print(timing[["algorithm", "seconds", "queries"]])
```

Use the same input and output transforms as in the corresponding serial example. Multiprocessing uses `spawn`; the model factory and the driver are importable Python modules.

## 3. Tune batching

| Parameter | Role |
|---|---|
| `max_batch_size` | Maximum sequence-scoring batch |
| `max_soft_batch_size` | Maximum gradient-scoring batch |
| `batch_wait_ms` | Request collection interval |

Set CPU thread counts before launching the experiment:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

`gpu_batches.csv/json` records batch sizes, model rows, worker composition, and scoring time. `algorithm_summary.csv` records each optimizer's runtime and query count. Parallel runtime includes queue waiting; `walltime.json` records the total elapsed time.

## Continue the same discovery and selection stages

The parallel backend returns the design table used by the remaining core functions:

```python
from sequence_optimization import call_motifs, select_candidates
scorer = make_scorer()
motifs = call_motifs(designs, model=scorer, objective=["maximize"], output_dir="outputs/parallel_motifs")
selected = select_candidates(designs, motifs=motifs, top_n=10, output_dir="outputs/parallel_panel")
```
