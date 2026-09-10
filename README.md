# Biosequence Optimization

`bioseq-optimization` is a model-agnostic DNA/RNA sequence-design toolkit.
It contains optimization algorithms, objectives, sequence initialization,
query accounting, and scorer interfaces. It does **not** contain model
architectures, training code, datasets, or checkpoints.

Ready-to-use Optimus 5-Prime, DeepSTARR, and other reproduced activity models
are maintained in the separate
[`bioseq-models`](https://github.com/UTR-CODE/bioseq-models) repository. A model
loaded there can be passed directly to every compatible optimizer here.

The workflow accepts single-output models (Optimus), multi-output models
(DeepSTARR), or independent models combined with `CombinedScorer`.
A direction list selects which outputs to raise, lower, or ignore.

## Installation

Install only the optimization package:

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-optimization.git
```

To use the ready-to-use models as well:

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-models.git
```

For development:

```bash
git clone git@github.com:UTR-CODE/bioseq-optimization.git
cd bioseq-optimization
python -m pip install -e .
python -m pytest -q
```

## Examples

### Single task: Optimus 5-Prime from `bioseq-models`

This example loads Optimus from `bioseq-models`, starts from five 50-nt 5′UTR
sequences, and returns the five best unique candidates found by the genetic
algorithm.

`model.score(sequences)` accepts nucleotide strings and returns a NumPy
prediction array; the loaded model handles its input conversion.

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize

model = load_model("optimus5prime", device="cpu")

initial = [
    "CCACTCGATTAACATGTTAACAACATACTCGTCCGGCCGATCAGCGGCTA",
    "CAAATCATGTGCAGCCCTGGCGACCGTACTGCGGTACAAGAAAGTATAGT",
    "GTTATACTAGAAGAAACTTGAGATTATGGAGCAGTCCGTCAAGGACGCGA",
    "CTTAGACAAAAACAACGCGCTTTCCAGTATGCGGAGCCTTGACGGTAAAC",
    "GGTACCTCTAGTTTAATCCCTCCCTTATACGGTGCAACCCGCTCCCCTCA",
]

result = genetic_algorithm_optimize(
    model,
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=5,
    budget=80,
    mutation_rate=0.1,
    elite_fraction=0.4,
    seed=7,
)

print(result.sequences[:5])
print(result.activities[:5])
```

No wrapper is required for Optimus with any of the nine black-box algorithms.
The runnable version is
[`sequence_optimization/examples/external_model.py`](sequence_optimization/examples/external_model.py).

`budget=80` means that this run may query the activity model for at most 80
candidate sequences. It is a model-evaluation budget, not a training epoch or
sequence count in the final result.

With the bundled checkpoint, the first five rows are:

| sequence | score | uncertainty | activity_mrl |
|---|---:|---:|---:|
| GTGAGATCAAGATAAATTTGTCCCTGTTTATCTCTAATCCACTGTCTAAA | 8.871757 | 0.000000 | 8.871757 |
| GAGAGAACTAGCTTAATTTGTCCCTGTTTACCTCTAATCTACTTTCTGAA | 8.739843 | 0.000000 | 8.739843 |
| GTGAGATCTAGATAAATTTGTCCCTGTTTACCTCTAATCCACTTCCTAAA | 8.737933 | 0.000000 | 8.737933 |
| GAGAGAACTAGATTAATTTGTCCCTGTTTACCTCTAATCGACTTTCTAAA | 8.715504 | 0.000000 | 8.715504 |
| GAGAGGACTATATTAATTTGTCGCTTTTTAACTCTAACCGACTTTCTAGA | 8.691080 | 0.000000 | 8.691080 |

The arrays are also available directly as `result.sequences`, `result.scores`,
`result.uncertainties`, `result.activities`, and `result.task_names`.

### Multi-task: one direction per output

Pass `objective` as a list in the same order as the model's output columns:

- `"maximize"`: make this output higher.
- `"minimize"`: make this output lower.
- `"ignore"`: exclude this output from optimization.

For any three-output scorer:

```python
from sequence_optimization import basesearch_optimize

result = basesearch_optimize(
    scorer,                          # scorer.score(sequences) returns [N, 3]
    length=249,       # choose the length accepted by your scorer
    objective=["maximize", "minimize", "ignore"],
    num_candidates=4,
    budget=100,
    seed=7,
)
print(result.activities)             # retains all three model outputs
```

The scorer returns `[N, outputs]`, with one direction per output.
Gradient algorithms additionally require
`score_soft(probabilities)` with gradients preserved.

### DeepSTARR example

DeepSTARR predicts `[developmental, housekeeping]` for 249-nt sequences.

```python
from bioseq_models import load_model
from sequence_optimization import basesearch_optimize

model = load_model("deepstarr", device="cpu")

result = basesearch_optimize(
    model,
    length=249,
    objective=["maximize", "maximize"],  # both high
    num_candidates=4,
    budget=100,
    seed=7,
)
print(result.to_frame())
```

Change only the list to choose another goal:

```python
objective = ["maximize", "minimize"]  # developmental high, housekeeping low
objective = ["minimize", "maximize"]  # developmental low, housekeeping high
objective = ["maximize", "ignore"]    # developmental high, housekeeping unconstrained
```

The list must match the number of outputs and contain at least one active task.
The default score is the minimum signed prediction across active tasks
(positive for maximize, negative for minimize), prioritizing the weakest task.
Ignored outputs remain in the results but do not affect ranking or gradients.
These are optimization directions, not absolute activity guarantees.
For different output scales or explicit thresholds, the optional
`MultiTaskObjective(directions=[...], scales=[...], ...)` interface is available.

[Python example](sequence_optimization/examples/deepstarr_multitask.py).

### Multiple models: lentiMPRA

Combine single-output scorers in output order. Raise K562 while lowering HepG2
and WTC11:

```python
from bioseq_models import load_model
from sequence_optimization import CombinedScorer, basesearch_optimize

scorer = CombinedScorer(
    [load_model(name, device="cpu") for name in
     ["agarwal_k562", "agarwal_hepg2", "agarwal_wtc11"]],
    names=["K562", "HepG2", "WTC11"],
)
result = basesearch_optimize(
    scorer, length=230,
    objective=["maximize", "minimize", "minimize"],
    num_candidates=4, budget=100, seed=7,
)
print(result.to_frame())
```

### Multiple models: DeepSTARR Mouse

Use the same optimizer to raise heart activity and lower limb and midbrain
activity. These are tissue models:

```python
scorer = CombinedScorer(
    [load_model(name, device="cpu") for name in
     ["deepstarr_mouse_heart_activity", "deepstarr_mouse_limb_activity",
      "deepstarr_mouse_midbrain_activity"]],
    names=["heart", "limb", "midbrain"],
)
result = basesearch_optimize(
    scorer, length=1001,
    objective=["maximize", "minimize", "minimize"],
    num_candidates=4, budget=100, seed=7,
)
print(result.to_frame())
```

[Python example functions](sequence_optimization/examples/multiple_models.py).
[Example sequences and predictions](docs/multiple_model_examples_2026-09-09.json)
were generated with a 100-query budget. The lentiMPRA best candidate predicts
K562/HepG2/WTC11 = 0.603/-0.585/-0.936. The Mouse short run predicts
heart/limb/midbrain = 0.022/0.041/0.032 and does not establish heart specificity.
Each child scorer must accept the
same designed sequence and handle its own encoding, padding or fixed context.
Gradient algorithms require every child to expose a differentiable `score_soft`;
wrap models individually before combining them.

`budget=100` counts combined candidate evaluations: each of three models gets
up to 100 sequence evaluations (300 total, excluding additional gradient
evaluations and internal ensemble members). Cumulative counts are available as
`scorer.model_query_counts` and `scorer.model_soft_query_counts`.
Cache hits can consume proposal budget without another model forward pass.
All raw outputs remain in the result. To adjust different output scales, use
fixed reference scales with `MultiTaskObjective(directions=[...], scales=[...])`.
These examples optimize raw predictions; search directions do not guarantee
absolute activity levels.

## Variable biological length

The sequence being designed may be shorter than the model input. Set
`pad_to_length` and choose the padding side on the scorer. The optimizer still
searches only the biological ACGT sequence; `N` is added only immediately
before model encoding.

```python
from bioseq_models import load_model
from sequence_optimization import genetic_algorithm_optimize

model = load_model(
    "optimus5prime",
    pad_to_length=50,
    padding_side="left",       # or "right"
)

result = genetic_algorithm_optimize(
    model,
    length=40,  # optimize the real 40-nt sequence
    num_candidates=16,
    budget=160,
    seed=7,
)
```

For discrete scoring, padding produces strings such as
`NNNNNNNNNNACGT...`. Model encoders should encode `N` as an all-zero
nucleotide vector. For differentiable optimization the toolkit adds the same
all-zero vectors to the soft one-hot tensor, so left/right padding behaves the
same for black-box and gradient algorithms.

RNA input is accepted, but `U` is always normalized to `T`. Internally and in
all returned candidates the alphabet is ACGT, never ACGU.

## Algorithms

Choose an algorithm using its function below. See
[model input requirements](docs/model_input_requirements.md)
for model-specific gradient adapters.

| Function | Algorithm | Optimus usage |
|---|---|---|
| `random_optimize` | Random search | Direct |
| `basesearch_optimize` | BaseSearch priority-decayed mutation search | Direct |
| `ism_optimize` / `ISM_optimize` | Greedy in-silico mutagenesis | Direct |
| `simulated_annealing_optimize` | Simulated annealing | Direct |
| `genetic_algorithm_optimize` | Genetic algorithm | Direct |
| `adalead_optimize` | AdaLead-style local search | Direct |
| `beam_search_optimize` | Beam search | Direct |
| `cross_entropy_optimize` | Cross-entropy method | Direct |
| `cbas_optimize` | CbAS with an internal autoregressive generator | Direct |
| `vanilla_activation_maximization_optimize` | Activation maximization | Differentiable wrapper |
| `fast_seqprop_optimize` | Fast SeqProp | Differentiable wrapper |
| `den_optimize` | Deep exploration network | Differentiable wrapper |
| `ledidi_optimize` | Ledidi-style objective maximization | Differentiable wrapper |

CbAS creates and fits its own proposal generator from the seed sequences and
queried activities. DEN also creates an internal generator, but it optimizes
that generator through the activity model, so the supplied scorer must support
`score_soft`.

For gradient algorithms, wrap the model to expose differentiable inputs:

```python
from sequence_optimization import DifferentiableSequencePredictor

differentiable_optimus = DifferentiableSequencePredictor(
    model.model,
    onehot=model.encoder,
    device=model.device,
    task_names=model.task_names,
    soft_input_transform=lambda probabilities: probabilities.transpose(1, 2),
)
```

The transpose adapts the optimizer's canonical `[batch, length, A/C/G/T]`
representation to Optimus's `[batch, A/C/G/T, length]` tensor layout. Pass
`differentiable_optimus` to any of the four gradient optimizers.

### BaseSearch

BaseSearch maintains a priority queue of parents, discounts a parent's priority
after selection, alternates broad and single-base mutations, and evaluates only
previously unseen sequences. It needs only discrete `score()` predictions.
The algorithm is based on
[Yao et al., Advanced Science (2026)](https://doi.org/10.1002/advs.202516600).

```python
from bioseq_models import load_model
from sequence_optimization import basesearch_optimize

model = load_model("optimus5prime", device="cpu")
result = basesearch_optimize(
    model, length=50, num_candidates=4, budget=100,
    basesearch_decay=0.995, mutation_rate=0.1, seed=7,
)
print(result.to_frame())
```

Search behavior and parameters:

- Start with the shared `initial` seeds or random initialization; maintain one
  queue entry per unique evaluated sequence.
- Alternate broad and local proposals, beginning with broad. Broad mutations
  choose `max(2, Binomial(length, mutation_rate))` distinct positions and replace
  every selected base. Length-one sequences always use a single substitution.
- Positive priorities decay by multiplication. For zero/negative activity,
  use `p - (1-decay) * max(abs(p), 1e-12)` to ensure selection lowers priority.
- With `basesearch_queue_size=1024`, prune to that size when the queue exceeds
  twice that size. Keep all scored sequences in the result archive and duplicate
  memory, even when removed from the parent queue.
- Stop on budget exhaustion, complete search-space coverage, or
  `basesearch_max_stalled=1024` consecutive duplicate proposals. The latter may
  stop before exhausting the budget; metadata records the reason.

Results rank by objective score. Queue priorities only control parent selection.

### Ledidi

The [Ledidi](https://github.com/jmschrei/ledidi)-style optimizer maximizes the
selected objective with an edit penalty:
`-mean(objective_score) + ledidi_edit_penalty * mean(number_of_substitutions)`.
It uses hard Gumbel-softmax samples and requires a differentiable scorer.

```python
from bioseq_models import load_model
from sequence_optimization import (
    DifferentiableSequencePredictor, ledidi_optimize,
)

model = load_model("optimus5prime", device="cpu")
scorer = DifferentiableSequencePredictor(
    model.model, onehot=model.encoder, device=model.device, task_names=model.task_names,
    soft_input_transform=lambda x: x.transpose(1, 2),
)
result = ledidi_optimize(
    scorer, length=50, num_candidates=4, budget=200,
    learning_rate=1.0, ledidi_edit_penalty=0.0, seed=7,
)
print(result.to_frame())
```

Supply `initial=[...]` to edit known sequences; otherwise random seeds are used.
`ledidi_edit_penalty` defaults to 0.1 per substitution; zero removes the explicit
edit penalty. `ledidi_temperature=1.0` and `ledidi_epsilon=1e-4` control sampling
temperature and the seed prior. Small epsilon strongly favors seed bases, so a
very short run may return unchanged seeds. Each seed has independent weights
and one sample per iteration. The result archive ranks candidates by activity;
the edit penalty guides search but is not a hard edit limit or a final ranking
criterion. `scores` contain the selected objective and `activities` retain all raw outputs.

Budget accounting follows the other gradient algorithms: repeated discrete
proposals count. Additional differentiable evaluations are reported separately
as `metadata['soft_query_count']`; `budget` is not the total number of network
forward/backward operations. Ledidi supports multi-output scorers through the
same objective interface as the other algorithms.

Use `run("algorithm_name", ...)` for the registry-based interface, or
`run_many(...)` to compare several algorithms with independent query budgets.

## Seed sequences or random initialization

Pass `initial=[...]` to any optimizer to use known sequences. If `initial` is
omitted, candidates are sampled uniformly at the requested length. The helper
`load_seed_sequences(path, length=50)` reads one sequence per line; blank lines and
lines starting with `#` are ignored.

## Stop at a plateau instead of choosing a fixed budget

`optimize_to_saturation` tracks the mean score of the retained candidates,
stops after a plateau, and returns the earliest candidate sets reaching chosen
fractions of the observed improvement.

```python
from sequence_optimization import SaturationConfig, optimize_to_saturation

saturated = optimize_to_saturation(
    model,
    length=50,
    algorithm="genetic_algorithm",
    num_candidates=16,
    initial=initial,
    saturation=SaturationConfig(
        fractions=(0.8, 0.9),
        maximum_queries=800,
    ),
    seed=7,
)

q80 = saturated[0.8]
q90 = saturated[0.9]
```

`maximum_queries` is only a safety ceiling. The normal stop condition is the
detected plateau. Here q80 and q90 mean 80% and 90% of the improvement from the
first complete candidate archive to the final observed plateau.

## Use your own model

See [Wrap your own model](docs/custom_models.md) for input conversion and
prediction examples.

## License and model provenance

This repository contains only optimization software. Consult `bioseq-models`
for checkpoint provenance, model-specific dependencies, and model licensing.
