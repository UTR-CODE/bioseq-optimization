# DREAM promoter optimization benchmark

The first DREAM promoter optimization run is complete: 12 weak-model
thresholds × 5 weak-model seeds × 8 optimizers = 480 conditions, with 100
110-bp promoter candidates per condition. Each condition uses one weak
checkpoint independently and the candidates are scored by the frozen strong
Autosome/LegNet DREAM model.

## Overall results

Across all 60 weak-model conditions:

| Optimizer | Mean strong score | Mean median score | Mean best score | Mean q90 queries |
|---|---:|---:|---:|---:|
| Vanilla activation maximization | 13.610 | 13.627 | 14.871 | 70,603 |
| DEN | 13.595 | 13.590 | 14.416 | 31,950 |
| Genetic algorithm | 13.580 | 13.587 | 14.025 | 12,290 |
| Fast SeqProp | 13.552 | 13.540 | 14.667 | 57,637 |
| Simulated annealing | 13.321 | 13.308 | 14.262 | 121,367 |
| AdaLead | 13.218 | 13.198 | 13.836 | 27,958 |
| ISM greedy | 13.108 | 13.110 | 13.941 | 194,250 |
| CbAS | 12.890 | 12.938 | 14.932 | 97,533 |

The best mean optimizer changes with the weak-model threshold:

| Threshold | Best mean strong-score method |
|---:|---|
| 1% | CbAS |
| 5% | Fast SeqProp |
| 10% | Vanilla AM |
| 20% | Genetic algorithm |
| 30% | Genetic algorithm |
| 40% | Vanilla AM |
| 50% | Genetic algorithm |
| 60% | Vanilla AM |
| 70% | ISM greedy |
| 80% | AdaLead |
| 90% | Vanilla AM |
| 95% | Genetic algorithm |

This threshold dependence means there is no universal winner. The highest
average strong scores occur at 80–95% visible training coverage, but individual
weak seeds still produce substantial variation.

## Protocol and caveat

The shared initialization in each condition is 50 top-label sequences from the
actual threshold fit split plus 50 uniformly random 110-mers. The q90 selection
rule chooses the earliest top-100 weak-score archive reaching 90% of the gain
from its first valid archive to its observed plateau. The run used at most
200,000 weak queries per optimizer and wrote one candidate/summary file per
condition.

This first implementation reports the eight optimization algorithms only; it
does not yet include Random-screening or Train-Top-100 controls. Also, the
current `top_1pct_count` field is computed against the 99th percentile *within
each returned set of 100 candidates*, not against a fixed global DREAM strong
score reference. It should therefore not be interpreted as a global top-1%
hit rate. Mean, median, best strong score and q90 query count are the valid
primary outputs for this run. A follow-up run should add fixed Random/Train-Top
controls and a global strong-reference percentile definition.

## Files

- `results/dream2022_optimization_benchmark/summary.csv`: 480 condition rows.
- `results/dream2022_optimization_benchmark/aggregate_summary.csv`: threshold ×
  optimizer aggregates.
- `results/dream2022_optimization_benchmark/bottom_*pct/run_*/`: candidate
  sequences, q90 metadata and shared initializations.
- `results/dream2022_optimization_benchmark/protocol.json`: exact run settings.

The output audit confirms 480 candidate files, 480 summary files, 48,000
candidate rows, 60 shared-initialization files, 100 candidates per condition,
and q90 query counts no larger than the 200,000 query ceiling.
