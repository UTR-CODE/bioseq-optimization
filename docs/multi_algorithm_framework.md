# Multi-Algorithm Sequence Optimization Framework

## Overview

This framework transforms the sequence_optimization package from a collection of individual optimization algorithms into a unified multi-algorithm exploration system for biological sequence design. Instead of running algorithms in isolation, the framework enables systematic exploration of the sequence design landscape through complementary optimization strategies, followed by unified quality control, solution-family discovery, and representative selection.

## Key Concept Shift

### Before (Algorithm Collection)
- User runs one algorithm at a time
- Each algorithm independently searches for top-N sequences
- User manually compares results across algorithms
- Focus: "Which algorithm is best?"

### After (Unified Exploration Framework)
- All algorithms run from shared initial population
- Results automatically merged into unified candidate pool
- Built-in analysis for convergence, diversity, and solution families
- Focus: "What design space regions exist and which are most promising?"

## Three Example Tasks

We provide complete, runnable examples for three representative biological sequence design scenarios:

### 1. Single-Task: Optimus 5-Prime MRL
**File:** `sequence_optimization/examples/01_optimus_mrl_suite.py`

- **Objective:** Maximize mean ribosome load (MRL)
- **Region:** 5'UTR (50 nucleotides)
- **Model:** Single output predictor
- **Use Case:** Optimizing translation initiation

```bash
python sequence_optimization/examples/01_optimus_mrl_suite.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000
```

### 2. Multi-Task: DeepSTARR Enhancer
**File:** `sequence_optimization/examples/02_deepstarr_multitask_suite.py`

- **Objective:** Minimize developmental, maximize housekeeping activity
- **Region:** Enhancer (249 nucleotides)
- **Model:** Two conflicting outputs
- **Use Case:** Cell-type specific enhancer design

```bash
python sequence_optimization/examples/02_deepstarr_multitask_suite.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000
```

### 3. Multiple Models: Agarwal lentiMPRA
**File:** `sequence_optimization/examples/03_agarwal_multimodel_suite.py`

- **Objective:** Maximize K562, minimize HepG2 and WTC11
- **Region:** Enhancer (230 nucleotides)
- **Models:** Three independent cell-type models
- **Use Case:** Cell-type specificity optimization

```bash
python sequence_optimization/examples/03_agarwal_multimodel_suite.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000
```

## Output Structure

Each example generates a comprehensive results directory:

```
results/01_optimus_mrl_suite/
├── config.json              # Complete run configuration
├── candidates.csv          # All candidates from all algorithms
├── summary.csv             # Per-algorithm performance metrics
├── initial.csv             # Shared initial population
├── genetic_algorithm.csv   # Per-algorithm results
├── basesearch.csv
├── beam_search.csv
├── ... (all 13 algorithms)
└── ledidi.csv
```

### Key Output Files

**candidates.csv** - Unified candidate pool with:
- `algorithm`: Source algorithm
- `rank`: Rank within that algorithm
- `sequence`: DNA sequence
- `score`: Scalarized objective score
- `activity_*`: Raw model predictions
- `query_count`: Optimizer queries
- `model_query_count`: Actual model calls (with shared caching)
- `elapsed_seconds`: Wall-clock time

**summary.csv** - Algorithm-level statistics:
- Performance metrics (best_score, mean_score)
- Resource usage (query_count, elapsed_seconds)
- Query efficiency (queries_per_second)
- Cache effectiveness (model_query_count vs query_count)

## New Module: CandidatePool

The `CandidatePool` class provides utilities for analyzing and processing the unified results:

```python
from sequence_optimization import CandidatePool, optimize_algorithms

# Run optimization suite
suite = optimize_algorithms(...)

# Create unified pool
pool = CandidatePool.from_suite_result(suite)

# Analyze convergence across algorithms
convergence = pool.get_convergence_summary()
# Shows which sequences were found by multiple algorithms

# Remove duplicates
unique_pool = pool.deduplicate(keep='best')

# Get top candidates across all algorithms
top_100 = pool.top_n_by_score(100)

# Export for downstream analysis
df = pool.to_frame()
```

## Shared Model Caching

A key efficiency feature is shared prediction caching across algorithms:

- **Black-box cache:** All 9 black-box algorithms share predictions
- **Gradient cache:** All 4 gradient algorithms share predictions
- **Result:** Later algorithms benefit from earlier computations

Example from test run:
- Genetic algorithm: 487 queries → 487 unique model calls
- BaseSearch: 500 queries → only 480 new model calls (20 were cached!)

## Algorithm Coverage

### Black-Box (9 algorithms)
1. `random` - Random search baseline
2. `ism_greedy` - Systematic single-mutation scan
3. `simulated_annealing` - Temperature-controlled exploration
4. `genetic_algorithm` - Crossover and mutation
5. `adalead` - Adaptive local exploration
6. `basesearch` - Priority queue search
7. `beam_search` - Breadth-first beam
8. `cross_entropy_method` - Distribution learning
9. `cbas` - Neural generative model

### Gradient-Based (4 algorithms)
10. `vanilla_activation_maximization` - Direct gradient ascent
11. `fast_seqprop` - Straight-through gradient
12. `den` - Noise-to-sequence generator
13. `ledidi` - Edit distance penalty

## Workflow Integration

### Stage 1: Multi-Algorithm Optimization (✓ Complete)
```python
from sequence_optimization import optimize_algorithms

suite = optimize_algorithms(
    model=model,
    algorithms=["genetic_algorithm", "basesearch", ...],
    length=50,
    initial=initial_sequences,
    objective=["maximize"],
    num_candidates=50,
    budget=5000,
)

pool = CandidatePool.from_suite_result(suite)
```

### Stage 2: Quality Control (Coming Soon)
```python
from sequence_optimization.qc import QualityControl

qc = QualityControl(reference_distribution=initial_sequences)
qc_results = qc.detect_ood(pool)
qc_results = qc.compute_uncertainty_scores(pool)
high_confidence = qc.filter_by_confidence(threshold=0.9)
```

### Stage 3: Solution Family Discovery (Coming Soon)
```python
from sequence_optimization.families import FamilyDiscovery

discovery = FamilyDiscovery()
families = discovery.cluster(pool, method='hierarchical', n_clusters=5)
motifs = discovery.extract_motifs(families, k=6)
convergence = discovery.analyze_algorithm_convergence(families)
```

### Stage 4: Representative Selection (Coming Soon)
```python
from sequence_optimization.representatives import select_representatives

recommendations = select_representatives(
    families=families,
    per_family=3,
    criteria=['score', 'centrality', 'novelty'],
)
```

## Running All Examples

Use the master script to run all three examples sequentially:

```bash
python sequence_optimization/examples/run_all_suites.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000 \
    --seed 42
```

Or select specific tasks:
```bash
python sequence_optimization/examples/run_all_suites.py \
    --tasks optimus deepstarr \
    --device cuda
```

## Performance Notes

**Test Results** (CPU, small scale):
- 2 algorithms, 20 initial, 10 candidates, 500 budget
- Genetic algorithm: 0.22 seconds
- BaseSearch: 0.06 seconds
- Total: < 1 second

**Expected Full-Scale** (GPU, realistic parameters):
- 13 algorithms, 100 initial, 50 candidates, 5000 budget
- Optimus (50nt): ~5-10 minutes on RTX 4090
- DeepSTARR (249nt): ~10-15 minutes on RTX 4090
- Agarwal (3 models, 230nt): ~15-25 minutes on RTX 4090

## Design Philosophy

### Multi-Algorithm Exploration
Rather than picking the "best" algorithm, use multiple complementary strategies:
- **Evolutionary:** genetic_algorithm, adalead
- **Systematic:** ism_greedy, beam_search
- **Stochastic:** simulated_annealing, random
- **Distribution:** cross_entropy_method, cbas
- **Gradient:** vanilla, fast_seqprop, den, ledidi

### Shared Resources
- Common initial population ensures fair comparison
- Shared model cache reduces redundant computation
- Unified scoring and reporting

### Downstream Analysis
- Convergence: Which sequences do multiple algorithms find?
- Diversity: What different solution regions exist?
- Reliability: Which candidates are consistently high-scoring?
- Interpretability: What motifs characterize each solution family?

## Next Steps

After running these examples, the generated candidate pools are ready for:

1. **Quality Control Analysis**
   - Identify out-of-distribution candidates
   - Flag low-confidence predictions
   - Check constraint violations

2. **Solution Family Discovery**
   - Cluster candidates by sequence similarity
   - Extract consensus motifs per family
   - Analyze algorithm convergence patterns

3. **Representative Selection**
   - Choose diverse, high-scoring candidates
   - Balance predicted performance with novelty
   - Generate experimental validation recommendations

These capabilities will be added in upcoming modules: `qc.py`, `families.py`, and `representatives.py`.

## References

- Main README: [README.md](../README.md)
- Algorithm details: [docs/algorithms.md](../docs/algorithms.md)
- Model tasks: [docs/models/](../docs/models/)
