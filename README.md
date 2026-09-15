# Biosequence Optimization

Design DNA/RNA sequences with a predictor and an optimization objective.

## Installation

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-optimization.git
```

For pretrained models used in the task guides:

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-models.git
```

## Algorithms

| Algorithm | Function | Predictor interface |
|---|---|---|
| [random](docs/algorithms/random.md) | `random_optimize` | `score` |
| [genetic_algorithm](docs/algorithms/genetic_algorithm.md) | `genetic_algorithm_optimize` | `score` |
| [ism_greedy](docs/algorithms/ism_greedy.md) | `ism_optimize` | `score` |
| [simulated_annealing](docs/algorithms/simulated_annealing.md) | `simulated_annealing_optimize` | `score` |
| [adalead](docs/algorithms/adalead.md) | `adalead_optimize` | `score` |
| [beam_search](docs/algorithms/beam_search.md) | `beam_search_optimize` | `score` |
| [cross_entropy_method](docs/algorithms/cross_entropy_method.md) | `cross_entropy_optimize` | `score` |
| [basesearch](docs/algorithms/basesearch.md) | `basesearch_optimize` | `score` |
| [cbas](docs/algorithms/cbas.md) | `cbas_optimize` | `score` |
| [vanilla_activation_maximization](docs/algorithms/vanilla_activation_maximization.md) | `vanilla_activation_maximization_optimize` | `score_soft` |
| [fast_seqprop](docs/algorithms/fast_seqprop.md) | `fast_seqprop_optimize` | `score_soft` |
| [den](docs/algorithms/den.md) | `den_optimize` | `score_soft` |
| [ledidi](docs/algorithms/ledidi.md) | `ledidi_optimize` | `score_soft` |

[Algorithm and model parameters](docs/algorithms.md).

## 1. Single task: custom CNN

The example weights measure nucleotide composition; use trained weights for a biological task.

```python
import torch
from torch import nn
from sequence_optimization import (
    SequencePredictor,
    genetic_algorithm_optimize,
    random_seed_sequences,
)


class SingleOutputCNN(nn.Module):
    def __init__(self, base="G"):
        super().__init__()
        self.conv = nn.Conv1d(4, 1, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, "ACGT".index(base), 0] = 1
            self.conv.bias.fill_(-0.25)

    def forward(self, values):
        return self.conv(values).mean(dim=2)


single_model = SequencePredictor(SingleOutputCNN(), device="cpu")
initial = random_seed_sequences(length=20, count=4, seed=7)
single_result = genetic_algorithm_optimize(
    single_model,
    length=20,
    initial=initial,
    objective=["maximize"],  # or [0.6] to stop after reaching score 0.6
    num_candidates=4,
)
print(single_result.to_frame())
```

| sequence | score | activity_task_0 |
|---|---:|---:|
| GGGGGGGGGGGGGGATGGGG | 0.65 | 0.65 |
| AGGGATGCGGGGGGACGGGG | 0.45 | 0.45 |
| GGGGACGCCGGGGGTAGGGG | 0.45 | 0.45 |
| GCCGGCGCAGGGGGTAGGCT | 0.30 | 0.30 |

A numeric entry sets a minimum raw-output target. A target-only search stops
when one scored sequence reaches the target (or one sequence reaches every
numeric target):

```python
initial = random_seed_sequences(length=20, count=4, seed=7)
target_result = genetic_algorithm_optimize(
    single_model,
    length=20,
    initial=initial,
    objective=[0.60],
    num_candidates=4,
)
print(target_result.to_frame())
```

Numeric targets can share a list with directions, for example
`objective=[0.60, "maximize", "ignore"]`. After a numeric target is reached,
later rounds select from candidates at or above that target while continuing
to optimize the remaining `maximize` or `minimize` outputs.

## 2. Multi-task: custom three-output CNN

One model returns three columns. Here they measure G, C and T composition,
respectively. Raise the first, lower the second and leave the third out of the
optimization objective:

```python
import torch
from torch import nn
from sequence_optimization import (
    SequencePredictor,
    genetic_algorithm_optimize,
    random_seed_sequences,
)


class ThreeOutputCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 3, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 2, 0] = 1
            self.conv.weight[1, 1, 0] = 1
            self.conv.weight[2, 3, 0] = 1
            self.conv.bias.fill_(-0.25)

    def forward(self, values):
        return self.conv(values).mean(dim=2)


multi_model = SequencePredictor(
    ThreeOutputCNN(),
    task_names=["G_activity", "C_activity", "T_activity"],
    device="cpu",
)
initial = random_seed_sequences(length=20, count=4, seed=7)
specific_result = genetic_algorithm_optimize(
    multi_model,
    length=20,
    initial=initial,
    objective=["maximize", "minimize", "ignore"],
    num_candidates=4,
)
print(specific_result.to_frame())
```

| sequence | score | G_activity | C_activity | T_activity |
|---|---:|---:|---:|---:|
| GGGGGGGGGGGTGGGAGGGG | 0.45 | 0.65 | -0.25 | -0.20 |
| GGGGGAGGGAGGTGAAGGGG | 0.375 | 0.50 | -0.25 | -0.20 |
| GGGGGGTGGAGGGGTATGGG | 0.375 | 0.50 | -0.25 | -0.10 |
| GGGGAGGAGAGGTGTAAGGG | 0.325 | 0.40 | -0.25 | -0.15 |

Entries follow output order. `maximize` raises an output, `minimize` lowers it,
and `ignore` excludes it from the objective. Ignored outputs are still reported
in `result.activities`; they are not constrained to remain unchanged.

To raise all three outputs, run a separate search from the same initial sequences:

```python
initial = random_seed_sequences(length=20, count=4, seed=7)
all_high_result = genetic_algorithm_optimize(
    multi_model,
    length=20,
    initial=initial,
    objective=["maximize", "maximize", "maximize"],
    num_candidates=4,
)
print(all_high_result.to_frame())
```

| sequence | score | G_activity | C_activity | T_activity |
|---|---:|---:|---:|---:|
| TCCCGCTCCGGGGGTCGGCT | 0.083 | 0.15 | 0.15 | -0.05 |
| TGGTGTTAACCTTACTATAC | 0.000 | -0.10 | -0.05 | 0.15 |
| CATATGAACAAGTCTTTGCG | -0.017 | -0.05 | -0.05 | 0.05 |
| CCCATAAATGTAGCCAGTGA | -0.033 | -0.05 | 0.00 | -0.05 |

The direction-list score is the equal-weight mean of the signed active outputs.
`ignore` outputs are excluded from that mean.

## 3. Multiple independent models: custom CNNs

```python
from sequence_optimization import (
    CombinedScorer,
    SequencePredictor,
    genetic_algorithm_optimize,
    random_seed_sequences,
)

model_a = SequencePredictor(SingleOutputCNN("G"), device="cpu")
model_b = SequencePredictor(SingleOutputCNN("C"), device="cpu")
model_c = SequencePredictor(SingleOutputCNN("T"), device="cpu")
combined = CombinedScorer(
    [model_a, model_b, model_c],
    names=["model_a", "model_b", "model_c"],
)
initial = random_seed_sequences(length=20, count=4, seed=7)
combined_result = genetic_algorithm_optimize(
    combined,
    length=20,
    initial=initial,
    objective=["maximize", "minimize", "minimize"],
    num_candidates=4,
)
print(combined_result.to_frame())
print(combined.model_query_counts)
```

| sequence | score | model_a | model_b | model_c |
|---|---:|---:|---:|---:|
| GGTGGGGGGGGGGGAAGGGG | 0.350 | 0.60 | -0.25 | -0.20 |
| GGGGGATAGGGGGGACGGGG | 0.300 | 0.50 | -0.20 | -0.20 |
| GGCGGAGAAGGGGGCAGGGG | 0.283 | 0.45 | -0.15 | -0.25 |
| GTGGAAGAAGTGGGCAGTGA | 0.183 | 0.25 | -0.20 | -0.10 |

The query counters for this run are
`{'model_a': 88, 'model_b': 88, 'model_c': 88}`.

`SingleOutputCNN` is defined in the first example. Each model evaluates the same
sequence and returns one output; each can use its own input conversion. The
objective list follows the order in `CombinedScorer`, so this search raises
model A's prediction and lowers B's and C's.

Each uncached candidate is sent to all three models. The counters above report
cumulative discrete sequence evaluations per model.

Use `objective=["maximize", "maximize", "maximize"]` for an all-high search,
or `ignore` to exclude a model's output from the objective. An ignored model is
still evaluated.

## Model tasks

- **5′UTR** — [Optimize MRL by Optimus 5-Prime](docs/models/optimus.md)
- **5′UTR, 3′UTR, or CDS** — [Optimize translation output by UTR-CODE](docs/models/utr_code.md)
- **CDS** — [Optimize CDS stability by CDS Stability](docs/models/cds_stability.md)
- **Enhancer** — [Optimize housekeeping activity and reduce developmental activity by DeepSTARR](docs/models/deepstarr.md)
- **Promoter** — [Optimize promoter activity by DREAM LegNet](docs/models/dream_legnet.md)
- **3′UTR** — [Optimize predicted mRNA stability by Cetnar](docs/models/cetnar_stability.md)
- **Enhancer** — [Optimize K562 specificity against HepG2 and WTC11 by Agarwal lentiMPRA](docs/models/agarwal.md)
- **Enhancer** — [Optimize heart specificity across mouse tissues by DeepSTARR Mouse](docs/models/deepstarr_mouse.md)
- **Enhancer** — [Optimize brain/VNC specificity across embryonic tissues by DeepSTARR-Embryo](docs/models/deepstarr_embryo.md)
- **Enhancer** — [Optimize K562 specificity against HepG2 and SK-N-SH by Malinois](docs/models/malinois.md)
- **CDS** — [Jointly optimize RPF and mRNA stability with synonymous codons](docs/tasks/joint_rpf_stability.md)
- **CDS** — [Optimize protein-preserving stability with synonymous codons](docs/tasks/synonymous_cds.md)

## Advanced features

### Results

An optimization result contains the designed sequences, scalarized objective
scores, raw predictions for every model output, query counts, and run metadata.

```python
print(single_result.to_frame())
print(single_result.sequences)
print(single_result.activities)
print(single_result.query_count, single_result.metadata["stop_reason"])
```

[Result fields and query accounting](docs/search_workflow.md#objectives-and-returned-data).

### Saturation screening

With `budget` omitted, every optimizer except random search detects an observed
plateau and returns the earliest candidate set that reaches 99% of the gain
from the initial archive to that plateau. To select other points, pass their
gain fractions in `SaturationConfig`; `budget` and `saturation` cannot be used
together. Random search uses `budget=10_000` by default and does not accept
`saturation`.

```python
from bioseq_models import load_model

from sequence_optimization import (
    SaturationConfig,
    genetic_algorithm_optimize,
    random_seed_sequences,
)

model = load_model("optimus5prime", device="cpu")
initial = random_seed_sequences(length=50, count=100, seed=7)
saturated = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=100,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9, 0.99),
        minimum_queries=1_000,
        plateau_patience_queries=2_000,
        maximum_queries=20_000,
        material_improvement=0.01,
    ),
    mutation_rate=0.02,
    elite_fraction=0.2,
    seed=7,
)
q80 = saturated[0.8]
q90 = saturated[0.9]
q99 = saturated[0.99]
print(q99.to_frame())
```

![Optimus 5-Prime saturation curve showing the initial, q80, q90, q99, and plateau points](docs/assets/optimus-saturation.png)

The selection fractions describe observed objective gain, not score
percentiles or guarantees of a global optimum. In this run, the top-100 mean
MRL rises from 6.44 initially to 9.66 at the observed plateau; q99 is the first
retained set reaching 99% of that gain.

[Saturation parameters and selection metadata](docs/search_workflow.md#stop-at-saturation).

### Sequence constraints

Pass one constraint function or a list through `constraint`. Each function
returns a pass probability in `[0, 1]`; only candidates passing every function
are sent to the model and admitted to the next round.

```python
import numpy as np


def gc_between_40_and_60(sequences):
    return np.asarray([
        0.40 <= (sequence.count("G") + sequence.count("C")) / len(sequence) <= 0.60
        for sequence in sequences
    ], dtype=float)


def excludes_homopolymers(sequence):
    return float(all(base * 5 not in sequence for base in "ACGT"))


initial = random_seed_sequences(length=20, count=4, seed=7)
constrained_result = genetic_algorithm_optimize(
    single_model,
    length=20,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    constraint=[gc_between_40_and_60, excludes_homopolymers],
)
print(constrained_result.to_frame())
```

[Copyable GC, motif, homopolymer, restriction-site, and soft constraints](docs/constraints.md).

### Protein-preserving CDS optimization

Set `is_CDS_optimized=True` with any optimizer to search synonymous codons.
Each initial CDS is optimized independently, and every returned candidate
translates to exactly the same amino-acid sequence as its corresponding
initial CDS.

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize

initial_cds = [
    "ATG" + "GCT" * 18 + "TAA",
    "ATG" + "GCT" * 17 + "GCC" + "TAA",
    "ATG" + "GCT" * 16 + "GCC" * 2 + "TAA",
    "ATG" + "GCT" * 15 + "GCC" * 3 + "TAA",
]
model = load_model("cds_stability", species="human", device="cpu")
cds_results = genetic_algorithm_optimize(
    model,
    length=len(initial_cds[0]),
    initial=initial_cds,
    objective=["maximize"],
    num_candidates=4,
    is_CDS_optimized=True,
)
for result in cds_results:
    print(result.to_frame())
```

Initial CDSs must start with `ATG`, end with `TAA`, `TAG`, or `TGA`, be in
frame, and contain no internal stop codon.

[Synonymous CDS requirements and independent results](docs/tasks/synonymous_cds.md).

### Fixed regions

Use `fixed_region` to prevent selected zero-based positions or Python-style
`start:stop` slices from changing.

```python
variable_regions = random_seed_sequences(length=15, count=4, seed=7)
initial = ["AAG" + sequence + "TT" for sequence in variable_regions]
fixed_result = genetic_algorithm_optimize(
    single_model,
    length=20,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    fixed_region=["0:3", "-2:"],
)
```

[Fixed-region coordinates and validation](docs/search_workflow.md#fixed-regions).

### Fixed query budget

Set a positive `budget` to use fixed query accounting instead of plateau
stopping. It must be at least `num_candidates`. Random search always uses a
fixed proposal budget; its default 10,000 includes the initial candidates and
all subsequent random proposals.

```python
initial = random_seed_sequences(length=20, count=4, seed=7)
budgeted_result = genetic_algorithm_optimize(
    single_model,
    length=20,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    budget=1_000,
)
```

[Budgets, caching, and model query counts](docs/search_workflow.md#candidate-count-and-budgets).

### Diversity threshold

`diversity` sets the minimum pairwise Hamming distance among returned
sequences. Set it to zero to disable diversity filtering.

```python
initial = random_seed_sequences(length=20, count=4, seed=7)
diverse_result = genetic_algorithm_optimize(
    single_model,
    length=20,
    initial=initial,
    objective=["maximize"],
    num_candidates=4,
    diversity=8,
)
```

[Candidate diversity behavior](docs/search_workflow.md#candidate-diversity).

### Custom model inputs

`SequencePredictor` supplies `[batch, 4, length]` A/C/G/T one-hot input by
default. For `[batch, N, length]` or another differentiable representation,
subclass it and override `prepare_input`.

```python
import torch

from sequence_optimization import SequencePredictor


class FiveChannelPredictor(SequencePredictor):
    def prepare_input(self, values):
        nucleotide_channels = values.transpose(1, 2)
        gc_channel = values[:, :, 1:3].sum(dim=2).unsqueeze(1)
        return torch.cat((nucleotide_channels, gc_channel), dim=1)


predictor = FiveChannelPredictor(five_channel_model, device="cuda")
```

[Default and custom PyTorch model inputs](docs/custom_models.md).
