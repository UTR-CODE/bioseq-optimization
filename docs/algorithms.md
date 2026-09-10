# Algorithm parameters

CbAS creates and fits its own proposal generator from the seed sequences and
queried activities. DEN also creates an internal generator, but it optimizes
that generator through the activity model, so the supplied scorer must support
`score_soft`.

For gradient algorithms, wrap the model to expose differentiable inputs:

```python
from bioseq_models import load_model
from sequence_optimization import DifferentiableSequencePredictor

model = load_model("optimus5prime", device="cpu")
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
