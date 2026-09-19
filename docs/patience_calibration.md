# Population-scaled plateau stopping

The September 2026 Optimus calibration changes default patience, not the
gradient algorithm, learning rate, initialization, or objective. It is an
empirical starting point for this checkpoint, not a universal convergence rule.

## Default rule

Let `P` be the initial search population and `K` the retained candidate count.
Use `N = max(P, K)` (without supplied seeds, `P = K`). For CDS optimization,
each reference has its own generated population; reference count is not P.

* Minimum counted queries: `10 * N`.
* Patience: `max(40 * N, 3 * B)` counted queries without a material improvement.
* Material improvement: `max(0.01, 0.001 * current score scale)` in the selected archive mean.
* Current suite/direct defaults return a diverse relaxed-knee set, with no
  implicit query ceiling. Historical measurements below used final-gain q90
  and the former suite ceiling `max(100000, 50 * N)`.
* Explicit saturation windows and fixed budgets override these defaults.
* Random search does not use plateau stopping; its current default is 100,000.

`B` is the nominal proposal batch. For gradient/population methods it is P;
for BaseSearch/CbAS it is the configured batch size; CEM uses the larger of its
batch size and K. For ISM it is P times the single-mutant neighborhood size;
for beam it is the beam width times that neighborhood size. Fixed regions and
alphabet size affect neighborhood size. Counting duplicates, actual batch
truncation and algorithm-specific stopping can affect actual query increments.

For gradient methods with P = K:

| N | Minimum queries | Patience queries | Approximate no-improvement updates |
|---:|---:|---:|---:|
| 20 | 200 | 800 | 40 |
| 100 | 1,000 | 4,000 | 40 |
| 200 | 2,000 | 8,000 | 40 |
| 1,000 | 10,000 | 40,000 | 40 |

The 1,000-population row is a formula illustration, not a measured run. With
100 seeds but only 20 retained candidates, use the 100-population row.
The minimum is a guard, not a patience-clock reset. Improvement restarts the
patience clock. A safety ceiling may stop a run before patience is exhausted.

## Experiments

`html_report_test/patience_benchmark.py` loads the real Optimus checkpoint on
CUDA and uses 50-nt initial sequences. It saves trajectories, final candidates,
and machine-readable comparison tables under gitignored `results/`.
Original `results/01_optimus_mrl_suite/candidates.csv` is not overwritten.

The initial grid covers all 12 non-random algorithms, P = K = 20 or 100, seeds
42 and 43, and a 10,000-query ceiling (48 trajectories). On identical recorded
trajectories it replays patience factors 2, 5, 10, 20, 40. Ten further runs
use P = K = 200 for four gradient methods and beam with a 20,000-query ceiling.
Four runs use P = 100, K = 20 for Vanilla and Ledidi. Extended 50,000-query
trajectories for all 12 non-random algorithms (P = K = 100, seed 42) check
whether the short-horizon result hides later improvements. In total this is
74 reference trajectories, plus two actual new-default gradient reruns.

Replay uses the same stochastic path and the same query ceiling; it does not
assume that changing the budget leaves an algorithm's schedule unchanged.
`recovered_gain` is improvement at the replayed stop divided by improvement at
the final observed point. It is not a naturalness metric or fraction of a
global optimum. The grid deliberately uses the old minimum N to isolate the
effect of patience; the new 10N guard can only delay those stop decisions.

Worst recovered gain across the initial four runs per algorithm:

| Algorithm | 2N | 10N | 20N | 40N |
|---|---:|---:|---:|---:|
| Vanilla activation maximization | 0.000 | 0.990 | 0.993 | 0.998 |
| Ledidi | 0.000 | 0.911 | 0.997 | 0.996 |
| Fast SeqProp | 0.967 | 0.985 | 0.989 | 0.992 |
| DEN | 0.908 | 0.955 | 0.981 | 0.981 |
| AdaLead | 0.717 | 0.956 | 0.979 | 1.000 |
| Genetic algorithm | 0.846 | 0.975 | 1.000 | 1.000 |
| Simulated annealing | 0.620 | 1.000 | 1.000 | 1.000 |
| Beam search | 0.477 | 0.477 | 0.477 | 1.000 |
| CbAS | 0.924 | 0.924 | 0.980 | 1.000 |
| Cross-entropy method | 0.952 | 0.952 | 1.000 | 1.000 |
| BaseSearch | 1.000 | 1.000 | 1.000 | 1.000 |
| ISM greedy | 1.000 | 1.000 | 1.000 | 1.000 |

Many 40N replays reach the ceiling: 1.000 does not prove convergence. Diversity
selection is greedy and its selected mean need not be monotone; slightly lower
gain at a longer patience is therefore possible. With a longer beam run,
40N alone stops at query 48,350, before gains near 50,000. This motivates the
three-proposal-batch floor: for K = 100, length 50, beam width 20, B = 3,000
and patience is 9,000 rather than 4,000. This floor is a conservative heuristic,
not proof that three unproductive batches imply convergence.

On the extended trajectories, replaying the new full rule retains 99.97% of
the observed gain for DEN and 99.78% for Fast SeqProp. Other algorithms reach
the end of the reference trajectory before this rule stops them. Those ends
include algorithm-internal stops and budget exhaustion, not just convergence.
The results therefore support avoiding premature stops, but do not establish
that 40N is a universally minimal or compute-optimal patience. The preserved
`default_policy_summary.csv` distinguishes replayed plateaus from trajectory
ends. The 50,000-query evidence is from one seed; the shorter grid uses two.

## Real default reruns

With the original 100 seeds and suite-equivalent algorithm seeds, the new
q90 selections have no original-seed overlap:

| Algorithm | Old median | New median | New best |
|---|---:|---:|---:|
| Vanilla | 7.115 | 8.797 | 9.080 |
| Ledidi | 7.115 | 8.922 | 9.121 |

These are q90 historical selections, not final-archive scores. Reproduction:

```bash
python html_report_test/patience_benchmark.py
python html_report_test/patience_benchmark.py --populations 100 --seeds 42 \
  --maximum 50000 --output results/patience_benchmark_long
```

Limitations: only one model, objective, sequence length and a small seed sample
were calibrated. Output count, diversity, model, initial population, learning
rate, neighborhood size and sequence length can all affect useful patience.
Future model-specific calibration may justify smaller windows, but two gradient
updates are demonstrably insufficient here.
