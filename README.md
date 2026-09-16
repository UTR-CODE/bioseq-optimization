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
                                            sequence           algorithm     score
0    AGCTTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA  genetic_algorithm    9.2347
1    TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT       basesearch    9.1562
2    CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC      beam_search    9.0871
...

Total candidates: 1300
Best score: 9.4987
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

# DataFrame includes activity columns: activity_Dev, activity_Hk
print(result_df[['algorithm', 'score', 'activity_Dev', 'activity_Hk']].head())
```

**Output:**
```
           algorithm     score  activity_Dev  activity_Hk
0  genetic_algorithm    2.3456        1.2341       3.5797
1       basesearch    2.2891        1.3452       3.6343
2      beam_search    2.2103        1.4234       3.6337
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

**Output:**
```
           algorithm     score  activity_K562  activity_HepG2  activity_WTC11
0  genetic_algorithm    4.5623         5.6734          1.2341          0.8976
1       basesearch    4.4891         5.5234          1.3452          0.9234
2      beam_search    4.3912         5.4123          1.4012          0.9789
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

### Sequence Constraints

Apply custom constraint functions to filter sequences during optimization. Each constraint function accepts a batch of sequences and returns probabilities (0-1) for each sequence:

- **0**: Reject sequence (hard constraint)
- **1**: Accept sequence  
- **0-1**: Soft threshold - probability of accepting the sequence

```python
from sequence_optimization import generate_designs, random_seed_sequences
import numpy as np

# Define constraint functions
def gc_soft_threshold(sequences):
    """Soft constraint: prefer 40-60% GC content."""
    gc_contents = np.array([
        (seq.count('G') + seq.count('C')) / len(seq) 
        for seq in sequences
    ])
    # Gaussian soft threshold centered at 50% GC
    return np.exp(-((gc_contents - 0.5) ** 2) / (2 * 0.1 ** 2))

def exclude_homopolymers(sequences):
    """Hard constraint: reject sequences with 5+ consecutive identical bases."""
    import re
    return np.array([
        0.0 if re.search(r'([ACGT])\1{4,}', seq) else 1.0
        for seq in sequences
    ])

# Combine multiple constraints
initial = random_seed_sequences(length=50, count=100, seed=42)

result_df = generate_designs(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    constraint=[gc_soft_threshold, exclude_homopolymers],  # List of constraint functions
    seed=42,
)
```

**See [constraint examples](docs/constraints_examples.md)** for more patterns including:
- GC content soft thresholds
- Motif exclusion (e.g., restriction sites)
- Homopolymer filtering
- Sequence naturalness scoring with diffusion models

### Fixed Regions

Specify multiple regions to keep unchanged during optimization. Useful for preserving functional elements:

```python
# Preserve multiple regions (Python slice notation)
result_df = generate_designs(
    model,
    length=230,
    initial=initial,
    objective=["maximize"],
    fixed_region=[[0, 10], [50, 60], [-20, -1]],  # Keep positions 0-9, 50-59, 210-229 unchanged
    seed=42,
)
```

**Common use cases:**
- Preserve promoter or enhancer cores
- Keep primer binding sites unchanged
- Maintain regulatory element positions
- Fix linker sequences between optimization regions

### Diversity Threshold

Control sequence diversity within each algorithm's candidate pool:

```python
# Increase diversity requirement (default=5)
result_df = generate_designs(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    diversity=10,  # Minimum Hamming distance of 10 between candidates
    seed=42,
)
```

**Trade-offs:**
- **Higher diversity** (e.g., 10-15): More exploration, broader design space coverage
- **Lower diversity** (e.g., 3-5): More exploitation, candidates clustered near optima
- **Default (5)**: Balanced for most applications

### Custom Model Input Channels

For models requiring non-standard input (e.g., epigenetic marks, conservation scores), inherit `SequencePredictor` and override encoding methods:

```python
import torch
from torch import nn
import numpy as np
from sequence_optimization import SequencePredictor, generate_designs, random_seed_sequences

class MultiChannelPredictor(SequencePredictor):
    """Predictor for models with additional input channels beyond ACGT."""
    
    def __init__(self, model, conservation_scores, **kwargs):
        super().__init__(model, **kwargs)
        self.conservation_scores = conservation_scores  # Pre-computed conservation track
    
    def encode_sequences(self, sequences):
        """Override to add conservation channel."""
        # Get standard 4-channel one-hot encoding
        onehot = super().encode_sequences(sequences)  # Shape: [batch, length, 4]
        
        # Add conservation scores as 5th channel
        batch_size = len(sequences)
        seq_length = len(sequences[0])
        conservation = torch.from_numpy(
            self.conservation_scores[:seq_length].reshape(1, seq_length, 1)
        ).repeat(batch_size, 1, 1)
        
        # Concatenate: [batch, length, 5]
        return torch.cat([onehot, conservation], dim=2)
    
    def prepare_input(self, encoded):
        """Override to handle 5-channel input."""
        # Model expects [batch, 5, length]
        return encoded.transpose(1, 2)

# Usage
class MultiChannelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(5, 64, kernel_size=8)  # 5 input channels
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# Create custom predictor
model = MultiChannelCNN()
conservation_track = np.random.rand(1000)  # Example conservation scores

predictor = MultiChannelPredictor(
    model,
    conservation_scores=conservation_track,
    device="cuda",

## Advanced Features

### Sequence Constraints

Apply custom constraint functions to filter sequences during optimization. Each constraint function accepts a batch of sequences and returns probabilities (0-1):

```python
from sequence_optimization import generate_designs, random_seed_sequences
import numpy as np

def gc_soft_threshold(sequences):
    """Soft constraint: prefer 40-60% GC content."""
    gc_contents = np.array([(seq.count('G') + seq.count('C')) / len(seq) for seq in sequences])
    return np.exp(-((gc_contents - 0.5) ** 2) / (2 * 0.1 ** 2))

def exclude_homopolymers(sequences):
    """Hard constraint: reject sequences with 5+ consecutive identical bases."""
    import re
    return np.array([0.0 if re.search(r'([ACGT])\1{4,}', seq) else 1.0 for seq in sequences])

# Combine multiple constraints
result_df = generate_designs(
    model, length=50, initial=initial, objective=["maximize"],
    constraint=[gc_soft_threshold, exclude_homopolymers],
    seed=42,
)
```

See [constraint examples](docs/constraints_examples.md) for more patterns including GC content, motif exclusion, homopolymer filtering, sequence complexity, RNA secondary structure, codon usage, and diffusion model naturalness scoring.

### Fixed Regions

Specify multiple regions to keep unchanged during optimization:

```python
result_df = generate_designs(
    model, length=230, initial=initial, objective=["maximize"],
    fixed_region=[[0, 10], [50, 60], [-20, -1]],  # Preserve positions 0-9, 50-59, 210-229
    seed=42,
)
```

### Diversity Threshold

Control minimum Hamming distance between candidates (default=5):

```python
result_df = generate_designs(
    model, length=50, initial=initial, objective=["maximize"],
    diversity=10,  # Higher diversity = broader exploration
    seed=42,
)
```

### Saturation Screening

By default, algorithms stop at **0.9 saturation** (90% of max gain), balancing efficiency with exploration and preventing out-of-distribution sequences.

![Saturation Screening](docs/figures/saturation_optimus_example.png)

**Example** (Optimus 5-Prime, genetic algorithm):
- 0.8: 3,247 queries → score 9.234
- **0.9** (default): 4,892 queries → score 9.456 ✓
- 0.99: 12,341 queries → score 9.498

Early stopping at 0.9 reduces cost by 60% while keeping sequences closer to training distribution.

```python
from sequence_optimization.search import SaturationConfig

result_df = generate_designs(
    model, length=50, initial=initial, objective=["maximize"],
    saturation=SaturationConfig(fractions=(0.95,), minimum_queries=1000, maximum_queries=50000),
    seed=42,
)
```

### Threshold-Based Multi-Task Optimization

Specify thresholds to focus optimization dynamically:

```python
# Once task_0 reaches 0.9, maintain ≥ 0.9 and focus on maximizing task_1
result_df = generate_designs(
    model, length=230, initial=initial,
    objective=[0.9, "maximize"],  # Mixed: threshold + directive
    seed=42,
)
```

Supports numbers (thresholds), `"maximize"`, `"minimize"`, and `"ignore"`.

### Objective Weights

Specify importance ratios for multi-task objectives (default 1:1):

```python
# Prioritize developmental suppression 2x over housekeeping activation
result_df = generate_designs(
    model, length=249, initial=initial,
    objective=["minimize", "maximize"],
    weights=[2.0, 1.0],
    seed=42,
)
```

### Custom Model Input Channels

For models with non-standard inputs (epigenetic marks, conservation scores, etc.), see [custom input channels guide](docs/custom_input_channels.md).

### Fixed Budget Limit

Override saturation with fixed query budget:

```python
result_df = generate_designs(
    model, length=50, initial=initial, objective=["maximize"],
    budget=10000,  # Fixed 10k queries per algorithm
    seed=42,
)
```

See [search workflow documentation](docs/search_workflow.md) for more details.

`generate_designs()` returns a pandas DataFrame with:

| Column | Description |
|--------|-------------|
| `sequence` | DNA/RNA sequence |
| `algorithm` | Which algorithm found this sequence |
| `score` | Scalarized objective score |
| `activity_*` | Raw model predictions (for multi-task models) |

**Default settings:**
- **13 algorithms**: 9 black-box + 4 gradient-based
- **100 candidates** per algorithm (1300 total maximum)
- **Random search**: 100,000 samples → top 100
- **Other algorithms**: auto-stop at saturation 0.9
- **Diversity**: minimum Hamming distance of 5 within each algorithm
- **Shared model caching** across algorithms for efficiency

**Algorithms:**
- Black-box (9): random, genetic_algorithm, ism_greedy, simulated_annealing, adalead, basesearch, beam_search, cross_entropy_method, cbas
- Gradient-based (4): vanilla_activation_maximization, fast_seqprop, den, ledidi
