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

All optimizer functions accept `length=...` directly; no sequence-space object
is needed. The default search alphabet is `ACGT`; optionally pass `alphabet`.
The length is the designed sequence length, before model-specific padding.

### Built-in onehot

```python
from sequence_optimization import onehot

values = onehot(["ACGT", "acgu"])  # NumPy float32, shape [2, 4, 4]: [N, L, channels]
values = onehot(["ACGT"], channels_first=True)  # [N, 4, L], for CNNs
```

Channel order defaults to A/C/G/T (`alphabet="AGCT"` changes the order).
`N` becomes all zeros, lowercase is accepted, and `U` maps to `T`.
Sequences must have equal lengths; other symbols are rejected.
Use `dtype="uint8"` for compact storage. `TorchSequenceScorer(model)` defaults
to float32 onehot in `[N, 4, L]` format, so no custom encoding function is needed.
Loaded bioseq-models models already provide their own input conversion.

The implementation uses NumPy byte lookup tables, referencing
[SeqPro's byte-based conversion](https://github.com/ML4GLand/SeqPro/blob/main/python/seqpro/_encoders.py).
It is implemented locally without requiring SeqPro or JIT compilation.

### Optional: a minimal custom CNN

This complete example defines a one-output CNN, wraps it as a scorer, and runs
the genetic algorithm. The fixed weights make the demonstration reproducible;
in an application, replace them with weights loaded from a trained checkpoint.

```python
import torch
from torch import nn

from sequence_optimization import (
    TorchSequenceScorer,
    genetic_algorithm_optimize,
)



class MinimalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Conv1d(4, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.convolution.weight.zero_()
            self.convolution.weight[0, 2, 0] = 1.0  # reward G

    def forward(self, inputs):
        return self.convolution(inputs).mean(dim=2)


scorer = TorchSequenceScorer(
    MinimalCNN(),
    device="cpu",
    task_names=("activity",),
)
result = genetic_algorithm_optimize(
    scorer,
    length=20,
    num_candidates=4,
    budget=40,
    mutation_rate=0.1,
    seed=7,
)
print(result.to_frame())
```

The runnable version is
[`sequence_optimization/examples/custom_cnn.py`](sequence_optimization/examples/custom_cnn.py).

### Single task: Optimus 5-Prime from `bioseq-models`

This example loads Optimus from `bioseq-models`, starts from five 50-nt 5′UTR
sequences, and returns the five best unique candidates found by the genetic
algorithm.

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

The optimizer does not depend on model names or task names. Replacing the model
requires a scorer returning `[N, outputs]`, the appropriate sequence length,
and a matching direction list. Gradient algorithms additionally require
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
All 13 algorithms are tested with the real DeepSTARR model in both-high and
both selective directions; small-budget tests check compatibility, not efficacy.

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
The combined scorer is independent of model names. Each child must accept the
same designed sequence and handle its own encoding, padding or fixed context.
Gradient algorithms require every child to expose a differentiable `score_soft`;
wrap models individually before combining them. All 13 algorithms are tested
with both three-model groups.

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

The [32-model single-output compatibility report](docs/single_task_model_compatibility_2026-09-09.md)
records real-checkpoint runs: all 32 models pass the eight black-box algorithms;
28 also pass the three gradient algorithms with wrappers (Cetnar on CPU).
That report predates Ledidi and BaseSearch. The remaining four need model-specific
differentiable adapters.

All algorithms use the same scorer, sequence space, initialization, objective,
and result contract.

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

All thirteen registered algorithms are exercised against the real Optimus
checkpoint in
[`tests/test_bioseq_models_integration.py`](tests/test_bioseq_models_integration.py).
The nine black-box algorithms accept the loaded model directly. For the four
gradient algorithms, expose the differentiable PyTorch network through the
existing wrapper:

```python
from sequence_optimization import DifferentiableTorchSequenceScorer

differentiable_optimus = DifferentiableTorchSequenceScorer(
    model.model,
    model.encoder,
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
This is a local implementation based on sections 4.15–4.18 of
[Yao et al., Advanced Science (2026)](https://doi.org/10.1002/advs.202516600),
not an execution of author-provided code.

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

Implementation choices where the article does not specify exact rules:

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

Results rank by original activity, not decayed queue priority. The mutation
schedule and negative-score extension are explicit adaptations; this is not a
claim of reproducing the paper's reported optimization performance.

### Ledidi for a single output

[Ledidi](https://github.com/jmschrei/ledidi) supports a single differentiable
output. The [mouse enhancer study](https://www.nature.com/articles/s41588-026-02729-1)
uses it with joint accessibility/activity models; that joint objective is not
required for single-model optimization.

Our local `ledidi` implementation uses Ledidi's `log(seed_one_hot + epsilon) + W`
parameterization and hard Gumbel-softmax samples. It minimizes
`-mean(activity) + ledidi_edit_penalty * mean(number_of_substitutions)`.
This is a maximization adaptation, not the official package's default MSE
target-matching objective or a reproduction of the paper's joint-model design.
It requires no additional package.

```python
from bioseq_models import load_model
from sequence_optimization import (
    DifferentiableTorchSequenceScorer, ledidi_optimize,
)

model = load_model("optimus5prime", device="cpu")
scorer = DifferentiableTorchSequenceScorer(
    model.model, model.encoder, device=model.device, task_names=model.task_names,
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

[Real-checkpoint smoke results](docs/ledidi_smoke_2026-09-09.json) cover Optimus
and DeepSTARR-Mouse heart activity. The earlier 32-model matrix does not yet
constitute Ledidi validation for every model.

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

Wrap a callable with `CallableScorer`, a frozen PyTorch model with
`TorchSequenceScorer`, or a differentiable PyTorch model with
`DifferentiableTorchSequenceScorer`. See the local
[`custom_cnn.py`](sequence_optimization/examples/custom_cnn.py) example for a
complete minimal integration.

A scorer must accept a batch of sequence strings and return either `[batch]`
or `[batch, tasks]` activities. The optimizer never trains the supplied
activity model.

## License and model provenance

This repository contains only optimization software. Consult `bioseq-models`
for checkpoint provenance, model-specific dependencies, and model licensing.
