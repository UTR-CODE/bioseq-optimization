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

Install MEME Suite for motif discovery and scanning; see [Installation](docs/installation.md).

## 1. Generate designs

Supply a trained PyTorch `model` mapping `[batch,4,length]` sequences to activities:

```python
from sequence_optimization import generate_designs, call_motifs, select_candidates

designs = generate_designs(
    model, length=200, objective=["maximize"],
    num_candidates=100, seed=42, output_dir="outputs/designs",
)
```

By default, 11 optimizers share 100 random starting sequences, run to a plateau, and retain 100 designs each at the 90% observed-gain stage. The result contains 1,100 source records with individual activity predictions and algorithm provenance. Runtime and query counts are recorded per algorithm.

## 2. Call motifs

```python
motifs = call_motifs(designs, model=model, output_dir="outputs/motifs")
```

The function uses the design objective to calculate contributions, extracts positive seqlets, discovers PWMs against dinucleotide-shuffled controls, and retains supported motifs.

## 3. Select candidates

With motifs, selection prioritizes motif composition diversity, matched motif-position diversity, and prediction score:

```python
gate = float(designs.drop_duplicates("sequence").score.quantile(0.9))
selected = select_candidates(
    designs, top_n=10, motifs=motifs, min_score=gate,
    output_dir="outputs/selected",
)
```

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
- [Motif composition and positional selection](docs/motif-diversity.md)
- [Algorithms and plateau stopping](docs/algorithms.md)
- [Parallel scoring backend](docs/parallel-scoring.md)
- [Evaluation](docs/validation.md)
