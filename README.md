# bioseq-optimization

Multi-algorithm optimization of biological sequences with motif composition and positional diversity.

A PyTorch sequence-to-activity model supplies the fitness landscape. Eleven optimization algorithms explore that landscape from a shared population. Their candidates are combined and selected using predicted activity, motif composition, and motif positions.

```text
PyTorch model + 100 starting sequences + optimization objective
                              ↓
              11 optimizers, each run to a plateau
                              ↓
                100 candidates per optimizer
                              ↓
          1,100 candidate records → unique sequences
                              ↓
            attribution → seqlets → motif catalog
                              ↓
            FIMO → motif presence and positions
                              ↓
    activity threshold → composition diversity → positional diversity
                              ↓
                      Top N sequences
```

## Installation

```bash
git clone https://github.com/UTR-CODE/bioseq-optimization.git
cd bioseq-optimization
python -m pip install -e '.[analysis,motifs]'
```

## Connect a PyTorch model

`model` is a trained `torch.nn.Module` that maps one-hot sequences `[batch, 4, length]` to activities `[batch, tasks]`. Wrap it once, then use the same interface for all optimization algorithms.

```python
from sequence_optimization import SequencePredictor, random_seed_sequences
from sequence_optimization.workflows import generate_candidates

scorer = SequencePredictor(model, tasks=["activity"], device="cuda")
initial = random_seed_sequences(200, 100, seed=42)

candidates, timing = generate_candidates(
    scorer, 200, initial, ["maximize"],
    output_dir="outputs/design", num_candidates=100,
)
```

The default workflow runs 11 algorithms to a plateau, retains 100 candidates per algorithm at the 90% observed-gain stage, and records runtime and query counts. Exact sequence duplicates are combined with their algorithm provenance. Random search is available as a separate reference.

`input_transform` adapts input layouts, fixed contexts, or embeddings. `output_transform` selects model outputs or constructs activity contrasts. Both transformations are shared by sequence scoring and gradient optimization.

## Three optimization settings

| Setting | Prediction model | Design objective | Tutorial |
|---|---|---|---|
| Single task | One model, one output | Maximize predicted activity | [Single-task design](docs/examples/single-task.md) |
| Multi-task | One model, multiple outputs | Target high, remaining outputs low | [Multi-task design](docs/examples/multi-task.md) |
| Multiple models | Several single-output models | Joint activity, weighted ensemble, or selective optimization | [Multiple-model design](docs/examples/multiple-models.md) |

Each tutorial shows model setup, objective definition, candidate generation, panel selection, and CSV/FASTA export.

Notebooks: [single task](examples/single-task.ipynb), [multi-task](examples/multi-task.ipynb), and [multiple models](examples/multiple-models.ipynb).

## Workflow and examples

| Document | Contents |
|---|---|
| [Installation](docs/installation.md) | Python package and optional motif tools |
| [PyTorch model interface](docs/pytorch-models.md) | Single- and multi-output models, input transforms, output transforms |
| [Candidate generation](docs/candidate-generation.md) | Shared seeds, 11 optimizers, plateau stopping, gain selection, timing |
| [Enhancer design](docs/examples/enhancer.md) | Cell-type specificity, motif diversity, activity and motif-position plots |
| [Promoter design](docs/examples/promoter.md) | Activity maximization, fixed flanks, diverse panels, contribution logos |
| [5′UTR design](docs/examples/5utr.md) | Translation optimization, diverse panels, upstream AUG annotation |
| [Motif discovery](docs/motif-discovery.md) | ISM, seqlets, STREME, motif support, catalog construction |
| [Motif diversity](docs/motif-diversity.md) | PWM similarity, Soft-Jaccard, Hungarian matching, hierarchical Max-Min |
| [Parallel optimization](docs/parallel-scoring.md) | CPU optimization workers and a shared batched scoring service |
| [Algorithms](docs/algorithms.md) | Algorithm names, gradient requirements, stopping parameters |
| [Evaluation](docs/validation.md) | Activity, specificity, diversity, and computational metrics |

Step-by-step notebooks: [enhancer](examples/enhancer.ipynb), [promoter](examples/promoter.ipynb), [5′UTR](examples/5utr.ipynb), and [motif selection with reference data](examples/motif_diversity.ipynb).
