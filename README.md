# bioseq-optimization

Sequence optimization with three core functions:

| Function | Purpose | Returns |
|---|---|---|
| `generate_designs` | Optimize a PyTorch sequence-to-activity model | Design table |
| `call_motifs` | Extract seqlets and discover motifs from the designs | Motif catalog |
| `select_candidates` | Select a diverse, high-quality panel | Selected design table |

```text
PyTorch predictor → generate_designs → 1,100 candidate records
                                      ↓
                                  call_motifs
                                      ↓
                              select_candidates → Top N
```

`select_candidates` uses Hamming Max-Min by default. Supplying `motifs` switches selection to motif composition and positional diversity.

## Installation

```bash
git clone https://github.com/UTR-CODE/bioseq-optimization.git
cd bioseq-optimization
python -m pip install -e '.[analysis,motifs]'
```

Install MEME Suite for official STREME motif discovery and FIMO scanning; see [Installation](docs/installation.md).

## 1. Generate designs

Define a CNN that maps one-hot DNA `[batch,4,length]` in A/C/G/T order to one activity value per sequence:

```python
import torch
from torch import nn
from sequence_optimization import (
    generate_designs, call_motifs, select_candidates, onehot,
)

class ActivityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, dna):
        return self.network(dna)  # [batch, 1] predicted activity
```

Train this network on DNA sequences and measured activities using a regression loss such as MSE. After training, save its weights with `torch.save(model.state_dict(), "activity_cnn.pt")`. Load the trained weights and predict activity from DNA strings:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ActivityCNN()
model.load_state_dict(torch.load("activity_cnn.pt", map_location="cpu", weights_only=True))
model = model.to(device).eval()

sequences = ["ACGT" * 50, "TGCA" * 50]  # Two 200-nt DNA sequences
x = torch.tensor(onehot(sequences, channels_first=True), device=device)
with torch.no_grad():
    activity = model(x)
print(activity)  # [2, 1]
```

Pass the trained model directly to the optimizer. The framework automatically wraps it with `SequencePredictor`, encodes DNA strings for discrete prediction, and implements `score_soft` for differentiable sequence probabilities. Your CNN only needs a differentiable `forward` method returning activity tensors; no custom `score` or `score_soft` methods are required.

```python
designs = generate_designs(
    model, length=200, objective=["maximize"],
    num_candidates=100, seed=42, output_dir="outputs/designs",
)
```

By default, 11 optimizers run in separate CPU processes and share 100 random starting sequences. Their scoring requests enter a shared queue; one scoring service combines compatible requests into batches for the model on its current device. Discrete prediction and gradient forward/backward requests use this service. Each optimizer runs to a plateau and retains 100 designs at the 90% observed-gain stage. The result contains 1,100 source records with individual activity predictions and algorithm provenance. Runtime and query counts are recorded per algorithm. `gpu_batches.csv/json` records batching and cross-worker utilization. See [Parallel scoring](docs/parallel-scoring.md) for batch-size controls.

Run these cells in a notebook. For a standalone Python script, place model loading and the workflow inside `if __name__ == "__main__":` so spawned CPU workers import only definitions.

## 2. Call motifs

The default backend uses official STREME for discovery and FIMO for seqlet support. It checks both executables before calculating contributions and provides an installation command when either is missing.

```python
motifs = call_motifs(designs, model=model, output_dir="outputs/motifs")
```

The function uses the design objective to calculate contributions, extracts positive seqlets, discovers PWMs with STREME against dinucleotide-shuffled controls, and retains motifs supported by at least ten seqlets from three parent sequences using FIMO matches. It returns a motif catalog and writes the DNA PWMs to **`outputs/motifs/catalog.meme`** in MEME format. You can also discover motifs with your own method and supply its MEME file directly to step 3.

## 3. Select candidates

With motifs, selection prioritizes motif composition diversity, matched motif-position diversity, and prediction score:

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.9))
selected = select_candidates(
    designs, top_n=10, motifs=motifs, min_score=gate,
    output_dir="outputs/selected",
)
```

Alternatively, pass a MEME file path. This can be the file exported in step 2 or a catalog produced by your own motif-discovery workflow:

```python
selected = select_candidates(
    designs, motifs="outputs/motifs/catalog.meme",  # Or "my_motifs.meme"
    top_n=10, min_score=gate, output_dir="outputs/selected_from_file",
)
```

The selector scans the supplied motifs onto the candidate sequences with FIMO and computes motif composition and positional diversity. With your own catalog, start directly at step 3; no model or attribution scores are needed for selection. MEME files contain DNA letter-probability matrices with `ALPHABET= ACGT` and unique motif IDs.

Without motifs, the same function selects by Hamming Max-Min:

```python
selected = select_candidates(designs, top_n=10, min_score=gate)
```

Both modes return a selected design table. `output_dir` exports CSV, FASTA, distances, and the selection trace.

## Design settings and examples

| Tutorial | Design objective |
|---|---|
| [Optimus 5-Prime: single task](docs/examples/single-task.md) | Maximize predicted MRL for 50-nt 5′UTRs |
| [DeepSTARR: multi-task](docs/examples/multi-task.md) | Increase developmental activity and decrease housekeeping activity |
| [Agarwal: multiple single-task models](docs/examples/multiple-models.md) | Increase K562 activity and decrease HepG2 and WTC11 activity |
| [Enhancer](docs/examples/enhancer.md) | Target-cell specificity and motif-diverse panels |
| [Promoter](docs/examples/promoter.md) | Activity optimization with fixed sequence contexts |
| [5′UTR](docs/examples/5utr.md) | Translation activity and upstream AUG annotation |

Each tutorial uses the same three core functions and has a corresponding [Notebook](examples).

## Methods and API

- [Three-function API](docs/core-api.md)
- [PyTorch model interface](docs/pytorch-models.md)
- [Candidate generation and timing](docs/candidate-generation.md)
- [Motif discovery](docs/motif-discovery.md)
- [Built-in discovery comparison](docs/motif-discovery-comparison.md)
- [Motif composition and positional selection](docs/motif-diversity.md)
- [Algorithms and plateau stopping](docs/algorithms.md)
- [Parallel scoring backend](docs/parallel-scoring.md)
- [Evaluation](docs/validation.md)
