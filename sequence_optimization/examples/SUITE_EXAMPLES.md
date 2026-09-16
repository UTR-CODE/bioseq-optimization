# Multi-Algorithm Optimization Suite Examples

This directory contains three complete examples demonstrating the unified multi-algorithm optimization framework for biological sequence design tasks.

## Overview

Each example runs all 13 available optimization algorithms (9 black-box + 4 gradient-based) on a specific sequence design task, generating a unified candidate pool for downstream quality control, clustering, and solution-family discovery.

## Examples

### 1. Optimus 5-Prime MRL (Single-Task)
**Script:** `01_optimus_mrl_suite.py`

- **Task:** Maximize mean ribosome load (MRL)
- **Model:** Optimus 5-Prime
- **Region:** 5'UTR (50 nucleotides)
- **Objective:** Single-task maximization

This demonstrates the simplest case: optimizing a single predictor output.

### 2. DeepSTARR Enhancer (Multi-Task)
**Script:** `02_deepstarr_multitask_suite.py`

- **Task:** Multi-task optimization
- **Model:** DeepSTARR
- **Region:** Enhancer (249 nucleotides)
- **Objective:** Minimize developmental activity, maximize housekeeping activity

This demonstrates multi-task optimization where different model outputs have conflicting objectives.

### 3. Agarwal lentiMPRA (Multiple Independent Models)
**Script:** `03_agarwal_multimodel_suite.py`

- **Task:** Cell type specificity
- **Models:** Three independent models (K562, HepG2, WTC11)
- **Region:** Enhancer (230 nucleotides)
- **Objective:** Maximize K562, minimize HepG2 and WTC11

This demonstrates combining multiple independent models with a specificity objective.

## Quick Start

### Prerequisites

```bash
# Install sequence_optimization package
cd /mnt/data/home/hexin/MPRA_benchmark
python -m pip install -e .

# Ensure bioseq-models is available
export PYTHONPATH="/home/hexin/bioseq-models:$PYTHONPATH"
```

### Run Individual Examples

```bash
# Example 1: Optimus MRL
python 01_optimus_mrl_suite.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000 \
    --seed 42

# Example 2: DeepSTARR
python 02_deepstarr_multitask_suite.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000 \
    --seed 42

# Example 3: Agarwal
python 03_agarwal_multimodel_suite.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000 \
    --seed 42
```

### Run All Examples at Once

```bash
python run_all_suites.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000 \
    --seed 42
```

## Command-Line Arguments

All scripts accept the following arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithms` | all 13 | Specific algorithms to run |
| `--initial-count` | 100 | Initial population size |
| `--num-candidates` | 50 | Candidates returned per algorithm |
| `--budget` | 5000 | Query budget per algorithm |
| `--model-batch-size` | 4096 | Model inference batch size |
| `--device` | cuda | Device (cuda/cpu) |
| `--weights-dir` | `/mnt/data/home/hexin/bioseq-models-weight-cache/0.10.0` | Model weights directory |
| `--output-dir` | `results/XX_*` | Output directory |
| `--seed` | 42 | Random seed |

### Example: Run Specific Algorithms Only

```bash
python 01_optimus_mrl_suite.py \
    --algorithms genetic_algorithm beam_search basesearch \
    --num-candidates 20 \
    --budget 2000
```

## Output Structure

Each example creates a dedicated results directory with the following files:

```
results/
├── 01_optimus_mrl_suite/
│   ├── config.json                  # Run configuration
│   ├── candidates.csv              # All candidates from all algorithms
│   ├── summary.csv                 # Per-algorithm performance summary
│   ├── initial.csv                 # Initial population sequences
│   ├── random.csv                  # Candidates from random search
│   ├── genetic_algorithm.csv       # Candidates from genetic algorithm
│   ├── ism_greedy.csv             # Candidates from ISM greedy
│   ├── simulated_annealing.csv    # Candidates from simulated annealing
│   ├── adalead.csv                # Candidates from AdaLead
│   ├── basesearch.csv             # Candidates from BaseSearch
│   ├── beam_search.csv            # Candidates from beam search
│   ├── cross_entropy_method.csv   # Candidates from CEM
│   ├── cbas.csv                   # Candidates from CbAS
│   ├── vanilla_activation_maximization.csv  # Gradient method
│   ├── fast_seqprop.csv           # Gradient method
│   ├── den.csv                    # Gradient method
│   └── ledidi.csv                 # Gradient method
├── 02_deepstarr_multitask_suite/
│   └── (same structure)
└── 03_agarwal_multimodel_suite/
    └── (same structure)
```

## Output Files Description

### `candidates.csv`
Combined results from all algorithms with columns:
- `algorithm`: Algorithm name
- `rank`: Rank within that algorithm (1 = best)
- `sequence`: DNA sequence
- `score`: Scalarized objective score
- `activity_*`: Raw model predictions for each task
- `query_count`: Total queries used by this algorithm
- `model_query_count`: Actual unique model calls (with caching)
- `elapsed_seconds`: Wall-clock time

### `summary.csv`
Per-algorithm statistics:
- `algorithm`: Algorithm name
- `initial_count`: Initial population size
- `candidate_count`: Returned candidates
- `query_count`: Total queries
- `model_query_count`: Unique model queries (shared cache)
- `elapsed_seconds`: Runtime
- `queries_per_second`: Throughput
- `best_score`: Top score achieved
- `mean_score`: Mean score of all candidates
- `stop_reason`: Why optimization stopped

### `config.json`
Complete run configuration for reproducibility

## Algorithm List

### Black-Box Algorithms (9)
1. `random` - Random search
2. `ism_greedy` - In silico mutagenesis greedy scan
3. `simulated_annealing` - Simulated annealing
4. `genetic_algorithm` - Genetic algorithm
5. `adalead` - AdaLead (adaptive local search)
6. `basesearch` - BaseSearch (priority queue)
7. `beam_search` - Beam search
8. `cross_entropy_method` - Cross-entropy method
9. `cbas` - Conditioning by Adaptive Sampling (neural)

### Gradient-Based Algorithms (4)
10. `vanilla_activation_maximization` - Direct gradient ascent
11. `fast_seqprop` - Fast SeqProp
12. `den` - Design by Efficient Noise (GAN-style)
13. `ledidi` - Ledidi (edit distance penalty)

## Performance Notes

- **GPU Memory:** All three tasks fit comfortably on a single GPU with 16GB+ VRAM
- **Runtime:** With budget=5000 and 13 algorithms:
  - Optimus (50nt): ~5-10 minutes on RTX 4090
  - DeepSTARR (249nt): ~10-15 minutes on RTX 4090
  - Agarwal (230nt, 3 models): ~15-25 minutes on RTX 4090
- **Shared Cache:** Model predictions are cached across algorithms, reducing redundant computation
- **Reproducibility:** Fixed seed ensures reproducible results

## Next Steps

The generated candidate pools are ready for:

1. **Quality Control**
   - Out-of-distribution detection
   - Uncertainty quantification
   - Constraint violation checking

2. **Solution Family Discovery**
   - Sequence clustering (Hamming distance, k-means)
   - Motif extraction and convergence analysis
   - Cross-algorithm comparison

3. **Representative Selection**
   - Select diverse, high-scoring candidates
   - Balance predicted performance with novelty
   - Recommend experimental validation set

These post-processing steps will be implemented in the `sequence_optimization.pool`, `sequence_optimization.families`, and `sequence_optimization.qc` modules.

## Troubleshooting

### CUDA Out of Memory
Reduce `--model-batch-size`:
```bash
python 01_optimus_mrl_suite.py --model-batch-size 2048
```

### Slow on CPU
Use smaller population and budget:
```bash
python 01_optimus_mrl_suite.py \
    --device cpu \
    --initial-count 20 \
    --num-candidates 10 \
    --budget 1000
```

### Model Weights Not Found
Specify weights directory:
```bash
python 01_optimus_mrl_suite.py \
    --weights-dir /path/to/bioseq-models-weight-cache/0.10.0
```

## References

- Optimus 5-Prime: [docs/models/optimus.md](../../docs/models/optimus.md)
- DeepSTARR: [docs/models/deepstarr.md](../../docs/models/deepstarr.md)
- Agarwal lentiMPRA: [docs/models/agarwal.md](../../docs/models/agarwal.md)
- Algorithm details: [docs/algorithms.md](../../docs/algorithms.md)
- Multi-optimizer suite: [docs/optimizer_suite.md](../../docs/optimizer_suite.md)
