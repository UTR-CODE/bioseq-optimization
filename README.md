# Biosequence Optimization

A unified multi-algorithm exploration framework for biological sequence design. Run 13 complementary optimization strategies from a shared initial population and obtain a unified candidate pool for downstream analysis.

[Framework overview figure will be placed here - Main Figure 1]

## Installation

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-optimization.git
```

For pretrained models:

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-models.git
```

## Complete Examples

### Example 1: Single-Task Optimization (Optimus 5-Prime MRL)

```python
import sys
sys.path.insert(0, "/home/hexin/bioseq-models")

from bioseq_models import load_model
from sequence_optimization import optimize_algorithms, random_seed_sequences

# Load model
model = load_model("optimus5prime", device="cuda", batch_size=256)

# Generate initial sequences
initial = random_seed_sequences(length=50, count=100, seed=42)

# Run all 13 algorithms
result_df = optimize_algorithms(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    seed=42,
)

# result_df is a pandas DataFrame with columns: sequence, algorithm, score
print(result_df.head(20))
print(f"\nTotal candidates: {len(result_df)}")
print(f"Algorithms: {result_df['algorithm'].unique()}")
```

**Run the example:**
```bash
python sequence_optimization/examples/01_optimus_mrl_suite.py --device cuda
```

### Example 2: Multi-Task Optimization (DeepSTARR)

```python
import sys
sys.path.insert(0, "/home/hexin/bioseq-models")

from bioseq_models import load_model
from sequence_optimization import optimize_algorithms, random_seed_sequences

# Load model
model = load_model("deepstarr", device="cuda", batch_size=128)

# Generate initial sequences
initial = random_seed_sequences(length=249, count=100, seed=42)

# Run all 13 algorithms
# Objective: minimize developmental activity, maximize housekeeping activity
result_df = optimize_algorithms(
    model,
    length=249,
    initial=initial,
    objective=["minimize", "maximize"],
    seed=42,
)

print(result_df.head(20))
```

**Run the example:**
```bash
python sequence_optimization/examples/02_deepstarr_multitask_suite.py --device cuda
```

### Example 3: Multiple Independent Models (Agarwal lentiMPRA)

```python
import sys
sys.path.insert(0, "/home/hexin/bioseq-models")

from bioseq_models import load_model
from sequence_optimization import (
    CombinedScorer,
    optimize_algorithms,
    random_seed_sequences,
)

# Load three cell-type models
model_k562 = load_model("agarwal_k562", device="cuda", batch_size=128)
model_hepg2 = load_model("agarwal_hepg2", device="cuda", batch_size=128)
model_wtc11 = load_model("agarwal_wtc11", device="cuda", batch_size=128)

# Combine models
combined = CombinedScorer(
    [model_k562, model_hepg2, model_wtc11],
    names=["K562", "HepG2", "WTC11"],
)

# Generate initial sequences
initial = random_seed_sequences(length=230, count=100, seed=42)

# Run all 13 algorithms
# Objective: maximize K562, minimize HepG2 and WTC11 (cell-type specificity)
result_df = optimize_algorithms(
    combined,
    length=230,
    initial=initial,
    objective=["maximize", "minimize", "minimize"],
    seed=42,
)

print(result_df.head(20))
```

**Run the example:**
```bash
python sequence_optimization/examples/03_agarwal_multimodel_suite.py --device cuda
```

## Algorithm Coverage

**Black-box (9):** random, genetic_algorithm, ism_greedy, simulated_annealing, adalead, basesearch, beam_search, cross_entropy_method, cbas

**Gradient-based (4):** vanilla_activation_maximization, fast_seqprop, den, ledidi

**Default Settings:**
- Each algorithm returns **100 candidates**
- **Random search**: samples 100,000 sequences and returns top 100 by predicted score
- **All other algorithms**: optimize until saturation plateau at **0.9** (90% of maximum gain)
- **Diversity control**: minimum Hamming distance of **5** between candidates within each algorithm

[Individual algorithm documentation](docs/algorithms/)

## Model-Specific Tasks

Pretrained models available via `bioseq-models`:

- [Optimus 5-Prime](docs/models/optimus.md) - 5′UTR MRL optimization
- [UTR-CODE](docs/models/utr_code.md) - Translation optimization
- [CDS Stability](docs/models/cds_stability.md) - CDS stability
- [DeepSTARR](docs/models/deepstarr.md) - Enhancer activity
- [DREAM LegNet](docs/models/dream_legnet.md) - Promoter activity
- [Cetnar](docs/models/cetnar_stability.md) - mRNA stability
- [Agarwal lentiMPRA](docs/models/agarwal.md) - Cell-type specificity
- [DeepSTARR Mouse](docs/models/deepstarr_mouse.md) - Tissue specificity (mouse)
- [DeepSTARR-Embryo](docs/models/deepstarr_embryo.md) - Tissue specificity (embryo)
- [Malinois](docs/models/malinois.md) - Cell-type specificity

## Custom Models

Use your own PyTorch CNN model:

```python
import torch
from torch import nn
from sequence_optimization import SequencePredictor, optimize_algorithms, random_seed_sequences

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=8)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        # x shape: (batch, 4, length) - one-hot encoded DNA
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# Create predictor
model = SimpleCNN()
predictor = SequencePredictor(model, device="cuda")

# Optimize
initial = random_seed_sequences(length=50, count=100, seed=42)
result_df = optimize_algorithms(
    predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    seed=42,
)

print(result_df.head(20))
```

See [custom models guide](docs/custom_models.md) for input format details.

## Sequence Constraints

Fix specific positions or enforce GC content bounds:

```python
from sequence_optimization import optimize_algorithms, random_seed_sequences

initial = random_seed_sequences(length=50, count=100, seed=42)

result_df = optimize_algorithms(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    fixed_region=(10, 20),  # Keep positions 10-19 unchanged
    gc_bounds=(0.4, 0.6),   # Enforce 40-60% GC content
    seed=42,
)
```

See [constraints documentation](docs/constraints.md) for motif requirements, edit distance limits, and more.

## CDS Optimization (Synonymous Mutations Only)

Optimize coding sequences while preserving the protein:

```python
from sequence_optimization import optimize_algorithms

# Your CDS sequence (must be divisible by 3)
cds_sequence = "ATGGCTAGC..." * 20  # 180 nt example

result_df = optimize_algorithms(
    model,
    length=len(cds_sequence),
    initial=[cds_sequence],
    objective=["maximize"],
    cds_mode=True,  # Enable codon-preserving mutations
    seed=42,
)

# All returned sequences encode the same protein
print(result_df.head(20))
```

See [CDS optimization guide](docs/cds_optimization.md) for details.

## Saturation Screening

By default, algorithms automatically stop when progress plateaus at the **0.9 saturation point** (90% of maximum observed gain). This balances exploration thoroughness with computational efficiency.

[Saturation screening figure will be placed here - showing 0.8, 0.9, 0.99 points with prediction values and query numbers]

**How it works:**
- Algorithm runs until improvement plateaus
- Automatically detects when 90% of maximum gain is reached
- Returns the candidate set at that saturation point
- Prevents unnecessary queries beyond diminishing returns

To customize saturation levels or use fixed query budgets, see [search workflow documentation](docs/search_workflow.md).

## Output Format

`optimize_algorithms()` returns a pandas DataFrame with:

| Column | Description |
|--------|-------------|
| `sequence` | DNA/RNA sequence |
| `algorithm` | Which algorithm found this sequence |
| `score` | Scalarized objective score |
| `activity_0`, `activity_1`, ... | Raw model predictions (for multi-task models) |

**Example output:**
```
     sequence                                          algorithm    score
0    AGCTTAGCTAGCTAGC...                              genetic_algorithm  9.234
1    TGCATGCATGCATGCA...                              basesearch        9.156
2    CGATCGATCGATCGAT...                              beam_search       9.087
...
```

Each algorithm contributes up to 100 candidates (1300 total maximum).
