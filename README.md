# Biosequence Optimization

A unified multi-algorithm framework for biological sequence design. Run 13 complementary optimization strategies and obtain diverse candidate sequences for experimental validation.

[Framework overview figure will be placed here - Main Figure 1 from the paper]

## Installation

```bash
# Install bioseq-optimization
python -m pip install git+https://github.com/UTR-CODE/bioseq-optimization.git

# Install pretrained models
python -m pip install git+https://github.com/UTR-CODE/bioseq-models.git
```

## Algorithms

| Algorithm | Type | Model interface |
|---|---|---|
| `random` | Black-box | `score()` |
| `genetic_algorithm` | Black-box | `score()` |
| `ism_greedy` | Black-box | `score()` |
| `simulated_annealing` | Black-box | `score()` |
| `adalead` | Black-box | `score()` |
| `basesearch` | Black-box | `score()` |
| `beam_search` | Black-box | `score()` |
| `cross_entropy_method` | Black-box | `score()` |
| `cbas` | Black-box | `score()` |
| `vanilla_activation_maximization` | Gradient | `score()` and `score_soft()` |
| `fast_seqprop` | Gradient | `score()` and `score_soft()` |
| `den` | Gradient | `score()` and `score_soft()` |
| `ledidi` | Gradient | `score()` and `score_soft()` |

`onehot()` and `SequencePredictor` use A/C/G/T channel order. A custom scorer's
`score()` method receives sequence strings, and `score_soft()` receives canonical
`[batch, length, A/C/G/T]` probabilities. Models with another channel order or
input representation perform their conversion inside those methods. A model that
only provides `score()` can be used by explicitly selecting black-box algorithms.

## Complete Examples

### Example 1: Single-Task Optimization (Optimus 5-Prime MRL)

Optimize 5'UTR sequences to maximize mean ribosome load (MRL):

```python
from bioseq_models import load_model
from sequence_optimization import generate_designs, random_seed_sequences

# Load model
model = load_model("optimus5prime", device="cuda", batch_size=256)

# Generate initial sequences
initial = random_seed_sequences(length=50, count=100, seed=42)

# Run all 13 algorithms
result_df = generate_designs(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    seed=42,
)

# result_df is a pandas DataFrame with columns: sequence, algorithm, score
print(result_df.head(20))
print(f"\nTotal candidates: {len(result_df)}")
print(f"Best score: {result_df['score'].max():.4f}")
```

**Output:**
```
                                            sequence algorithm     score  activity_mrl
0  TTGAGATTTTATTAATTTCTGAATTTTTAACTTTTAGATTTAAGAT...   adalead  9.523573      9.523573
1  TTGTGATTTTATTAATTTTTGATTTTTTAACTTTTAGAGTTAGAAT...   adalead  9.510443      9.510443
2  TAGTGTTTTGATTAATTTTTGAATTATTAACTTTTAGATTTAAGAT...   adalead  9.502789      9.502789
...

Total candidates: 1300
Best score: 9.5539
```

### Example 2: Multi-Task Optimization (DeepSTARR)

Optimize enhancer sequences for tissue specificity:

```python
from bioseq_models import load_model
from sequence_optimization import generate_designs, random_seed_sequences

# Load model
model = load_model("deepstarr", device="cuda", batch_size=128)

# Generate initial sequences
initial = random_seed_sequences(length=249, count=100, seed=42)

# Run all 13 algorithms
# Objective: minimize developmental activity, maximize housekeeping activity
result_df = generate_designs(
    model,
    length=249,
    initial=initial,
    objective=["minimize", "maximize"],
    seed=42,
)

# DataFrame includes activity columns for both model outputs
print(
    result_df[
        ["algorithm", "score", "activity_developmental", "activity_housekeeping"]
    ].head()
)
```

**Output:**
```
  algorithm     score  activity_developmental  activity_housekeeping
0   adalead  1.730678                1.192306               4.653661
1   adalead  1.729318                1.161735               4.620370
2   adalead  1.723419                1.374944               4.821783
```

### Example 3: Multiple Independent Models (Agarwal lentiMPRA)

Optimize for cell-type specificity across three cell lines:

```python
from bioseq_models import load_model
from sequence_optimization import CombinedScorer, generate_designs, random_seed_sequences

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
result_df = generate_designs(
    combined,
    length=230,
    initial=initial,
    objective=["maximize", "minimize", "minimize"],
    seed=42,
)

# DataFrame includes: activity_K562, activity_HepG2, activity_WTC11
print(result_df[['algorithm', 'score', 'activity_K562', 'activity_HepG2', 'activity_WTC11']].head())
```

The Agarwal adapters convert canonical A/C/G/T inputs to the A/G/C/T channel
order used by MPRALegNet in both scoring paths.

**Output:**
```
  algorithm     score  activity_K562  activity_HepG2  activity_WTC11
0   adalead  2.942906       6.026797       -1.372510       -1.429412
1   adalead  2.941358       5.981269       -1.401444       -1.441360
2   adalead  2.932116       6.054406       -1.337788       -1.404153
```

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
from sequence_optimization import SequencePredictor, generate_designs, random_seed_sequences

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
result_df = generate_designs(
    predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    seed=42,
)

print(result_df.head(20))
```

See [custom models guide](docs/custom_models.md) for input format details.

## Advanced Features

`generate_designs()` accepts optional arguments to constrain the search space and shape the
objective: sequence constraint functions, fixed regions, diversity thresholds, saturation
screening, threshold-based multi-task objectives, objective weights, CDS synonymous-codon
optimization, fixed query budgets, and custom model input channels.

See the [advanced features guide](docs/advanced_features.md) for details and examples.

## Output Format

`generate_designs()` returns a pandas DataFrame with:

| Column | Description |
|--------|-------------|
| `sequence` | DNA/RNA sequence |
| `algorithm` | Which algorithm found this sequence |
| `score` | Scalarized objective score |
| `activity_*` | Raw model predictions (for multi-task models) |

**Default settings:**
- **13 algorithms**: 9 black-box + 4 gradient-based
- **100 candidates** per algorithm
- **Random search**: 10,000 samples → top 100 by default
- **Other algorithms**: auto-stop at saturation 0.9
- **Diversity**: minimum Hamming distance of 5 within each algorithm
- **Shared model caching** across algorithms for efficiency

**Algorithms:**
- Black-box (9): random, genetic_algorithm, ism_greedy, simulated_annealing, adalead, basesearch, beam_search, cross_entropy_method, cbas
- Gradient-based (4): vanilla_activation_maximization, fast_seqprop, den, ledidi
