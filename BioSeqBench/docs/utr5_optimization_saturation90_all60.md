# 5′UTR optimization with all 60 weak models

## Protocol

This benchmark uses the 60 newly trained `WeakSeqCNN-v1` checkpoints
(12 lower-tail fractions × 5 seeds) independently. A run never averages weak
models: `run_00` uses `member_00`, ..., `run_04` uses `member_04` for that same
threshold.

The eight optimized methods are ISM greedy, simulated annealing, genetic
algorithm, AdaLead, vanilla categorical activation maximization, Fast SeqProp,
CbAS, and DEN. For each weak model and replicate, all eight methods receive the
same 100 starting sequences: the 50 highest-label sequences in that model's
actual fit split plus 50 uniformly sampled full-space random 50-mers. Random
screening and observed Train Top-100 are reported as controls.

There is no common query budget. Each optimizer records its monotone top-100
weak-score trajectory until a post-knee plateau (50,000 queries without a
material improvement) or the 840-second soft runtime limit. The final 100
sequences are the earliest archive snapshot reaching 90% of the observed gain
from the first valid archive to the final plateau. Random controls use the
per-condition median or mean q90 query count of the eight optimized methods.
The emergency query ceiling is 100,000,000; it was not reached in these runs.

Every final set contains 100 unique sequences outside the weak model's actual
fit split. Candidate sequences are evaluated only after generation with the
strong `Optimus5PrimePaper` model. The final run contains 660 conditions (60
weak models × 11 methods) and 66,000 strong-scored candidates.

## Overall strong-model results

Values are averaged over all 60 weak-model conditions. Train Top-100 is an
observed-data reference and is not an optimizer.

| Method | Mean strong score | Median strong score | Best strong score | Mean top-1% hits (per 100) | Mean reference percentile |
|---|---:|---:|---:|---:|---:|
| Fast SeqProp | 7.169 | 7.213 | 8.079 | 18.32 | 65.13 |
| Vanilla AM | 7.104 | 7.096 | 8.116 | 17.42 | 63.05 |
| Random mean-q90 | 7.058 | 7.117 | 8.287 | 5.13 | 61.61 |
| DEN | 7.054 | 7.061 | 7.760 | 10.22 | 62.79 |
| Random median-q90 | 7.050 | 7.101 | 8.239 | 4.33 | 61.45 |
| Simulated annealing | 7.036 | 7.099 | 8.187 | 4.68 | 61.10 |
| CbAS | 7.008 | 7.041 | 8.241 | 6.87 | 60.55 |
| AdaLead | 6.993 | 7.002 | 7.454 | 9.60 | 59.75 |
| Genetic algorithm | 6.915 | 6.911 | 7.346 | 8.93 | 59.32 |
| ISM greedy | 6.850 | 6.831 | 7.583 | 10.23 | 55.31 |
| Train Top-100 reference | 6.227 | 6.217 | 7.983 | 0.75 | 46.50 |

These averages combine thresholds with very different surrogate strengths, so
per-threshold results should be preferred for conclusions. The best mean method
varies by threshold: Random controls lead at 40–50%, DEN leads at 70%, Fast
SeqProp leads at 5%, 30%, and 90%, and vanilla AM leads at 20%, 80%, and 95%.
This is evidence against a universal optimizer winner.

## Runtime and query behavior

No optimized run exceeded 840 seconds. Mean runtime per optimized condition
(seconds) was: CbAS 56.3, DEN 25.2, vanilla AM 18.9, Fast SeqProp 18.4,
simulated annealing 9.0, genetic algorithm 4.5, ISM 4.4, and AdaLead 3.6.
Query counts are intentionally heterogeneous: GA and AdaLead often reached q90
in a few thousand queries, whereas ISM, CbAS, and some gradient runs required
hundreds of thousands or millions to establish a plateau. The q90 snapshot,
not the final plateau query count, determines the 100 reported sequences.

## Files

- `results/utr5_optimization_saturation90_all60/protocol.json`: exact protocol,
  initialization and q90/runtime rules.
- `results/utr5_optimization_saturation90_all60/summary.csv`: 660 condition
  summaries, including strong scores, q90 counts and weak checkpoint paths.
- `results/utr5_optimization_saturation90_all60/aggregate_summary.csv`:
  threshold × method aggregates.
- `results/utr5_optimization_saturation90_all60/all_candidates.csv`: all
  66,000 candidate-level strong scores.
- `results/utr5_optimization_saturation90_all60/all_shared_initial_sequences.csv`:
  all 60 × 100 shared initial sequences.
- `results/utr5_optimization_saturation90_all60/bottom_*pct/run_*/`:
  per-condition candidates, trajectories, summaries, and initialization files.

The final audit confirms 660 summaries, 66,000 candidate rows, 6,000 initial
sequences, 100 unique candidates per condition, exactly 50 train-top and 50
random starts per condition, no random-start membership in the weak fit pool,
and no runtime-limit truncations.
