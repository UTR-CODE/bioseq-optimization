# Biosequence Optimization

A unified multi-algorithm exploration framework for biological sequence design. Instead of running optimization algorithms in isolation, this framework systematically explores the sequence design landscape through 13 complementary optimization strategies, generating unified candidate pools for quality control, solution-family discovery, and representative selection.

## Framework Philosophy

Traditional approach: "Which single algorithm finds the best sequence?"

**Our approach**: "What design space regions exist, and which solution families are most promising?"

- Run all 13 algorithms from a shared initial population
- Merge results into a unified candidate pool with shared model caching
- Analyze convergence, diversity, and solution families
- Select diverse representatives for experimental validation

## Installation

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-optimization.git
```

For pretrained models:

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-models.git
```

## Quick Start

Run all 13 algorithms on three representative tasks:

```bash
# Install package
python -m pip install -e .

# Run all three example tasks
python sequence_optimization/examples/run_all_suites.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000
```

This generates unified candidate pools in `results/` for:
1. **Optimus 5-Prime MRL** (single-task, 50 nt)
2. **DeepSTARR enhancer** (multi-task, 249 nt)
3. **Agarwal lentiMPRA** (3 models, 230 nt)

## Multi-Algorithm Optimization Suite

The core workflow uses `optimize_algorithms()` to run multiple complementary optimization strategies from a shared initial population:

```python
from bioseq_models import load_model
from sequence_optimization import (
    CandidatePool,
    optimize_algorithms,
    random_seed_sequences,
)

# Load model and generate initial population
model = load_model("optimus5prime", device="cuda")
initial = random_seed_sequences(length=50, count=100, seed=42)

# Run all 13 algorithms
suite = optimize_algorithms(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=50,
    budget=5000,
    seed=42,
)

# Analyze unified candidate pool
pool = CandidatePool.from_suite_result(suite)
convergence = pool.get_convergence_summary()  # Sequences found by multiple algorithms
unique_pool = pool.deduplicate(keep='best')   # Remove cross-algorithm duplicates
top_100 = pool.top_n_by_score(100)            # Best candidates overall

# Export results
suite.save_csv("results/optimus_suite/")
```

### Complete Examples

- [01_optimus_mrl_suite.py](sequence_optimization/examples/01_optimus_mrl_suite.py) - Single-task (50nt)
- [02_deepstarr_multitask_suite.py](sequence_optimization/examples/02_deepstarr_multitask_suite.py) - Multi-task (249nt)
- [03_agarwal_multimodel_suite.py](sequence_optimization/examples/03_agarwal_multimodel_suite.py) - Multiple models (230nt)

### Documentation

- [Multi-Algorithm Framework Guide](docs/multi_algorithm_framework.md) - Complete framework overview
- [Suite Examples Guide](sequence_optimization/examples/SUITE_EXAMPLES.md) - Detailed usage
- [Quick Start Guide](QUICKSTART.md) - Get started immediately

## Algorithm Coverage

**Black-box (9):** random, genetic_algorithm, ism_greedy, simulated_annealing, adalead, basesearch, beam_search, cross_entropy_method, cbas

**Gradient-based (4):** vanilla_activation_maximization, fast_seqprop, den, ledidi

All algorithms support single-task, multi-task, multiple independent models, sequence constraints, and fixed regions.

### Individual Algorithm Documentation

For detailed documentation on each algorithm and single-algorithm usage:
- [Algorithm Overview](docs/algorithms.md)
- [Individual Algorithm Guides](docs/algorithms/)

## Supported Tasks

### Model-Specific Tasks
- **5′UTR** — [Optimus 5-Prime MRL optimization](docs/models/optimus.md)
- **5′UTR, 3′UTR, or CDS** — [UTR-CODE translation optimization](docs/models/utr_code.md)
- **CDS** — [CDS stability optimization](docs/models/cds_stability.md)
- **Enhancer** — [DeepSTARR activity optimization](docs/models/deepstarr.md)
- **Promoter** — [DREAM LegNet promoter optimization](docs/models/dream_legnet.md)
- **3′UTR** — [Cetnar mRNA stability optimization](docs/models/cetnar_stability.md)
- **Enhancer** — [Agarwal lentiMPRA cell specificity](docs/models/agarwal.md)
- **Enhancer** — [DeepSTARR Mouse tissue specificity](docs/models/deepstarr_mouse.md)
- **Enhancer** — [DeepSTARR-Embryo tissue specificity](docs/models/deepstarr_embryo.md)
- **Enhancer** — [Malinois cell specificity](docs/models/malinois.md)

### Advanced Tasks
- [Joint RPF and mRNA stability with synonymous codons](docs/tasks/joint_rpf_stability.md)
- [Protein-preserving stability optimization](docs/tasks/synonymous_cds.md)

## Additional Features

### Sequence Constraints
- Fixed regions (preserve specific positions)
- GC content bounds
- Motif requirements
- Edit distance limits

### CDS Optimization
- Protein-conserving codon optimization
- Synonymous mutations only
- Compatible with all algorithms

See [constraints documentation](docs/constraints.md) and [CDS guide](docs/cds_optimization.md) for details.

## Custom Models

The framework supports custom PyTorch models:

```python
from sequence_optimization import SequencePredictor

predictor = SequencePredictor(
    your_pytorch_model,
    device="cuda",
    task_names=["output1", "output2"],
)

suite = optimize_algorithms(
    predictor,
    length=100,
    initial=initial_sequences,
    objective=["maximize", "minimize"],
    num_candidates=50,
    budget=5000,
)
```

See [custom models guide](docs/custom_models.md) for details on model interfaces and input preparation.

## Advanced Features

### Saturation Screening

Stop optimization when observed progress plateaus instead of using a fixed budget:

```python
from sequence_optimization import SaturationConfig

suite = optimize_algorithms(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=50,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9),
        minimum_queries=40,
        plateau_patience_queries=80,
        maximum_queries=400,
    ),
)
```

See [search workflow documentation](docs/search_workflow.md) for details.

### Result Analysis

```python
# View summary statistics
print(suite.summary_frame())

# Access individual algorithm results
for algo, result in suite.results.items():
    print(f"{algo}: {result.query_count} queries, best score: {result.scores.max()}")

# Check model query counts (with caching)
print(suite.model_query_counts)
```

## Contributing

See [HANDOFF.md](HANDOFF.md) for development documentation and architecture details.

## License

[Add license information]

## Citation

[Add citation information]
