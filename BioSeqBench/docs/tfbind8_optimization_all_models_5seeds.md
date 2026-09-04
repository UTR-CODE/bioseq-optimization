# TFBind8 all-model, five-seed optimization benchmark

## Protocol

- Weak models: all 12 lower-tail CNN ensembles (1%, 5%, 10%, 20%, ..., 90%, 95%).
- Replicates: five independent deterministic seeds per model and optimizer.
- Optimizers: Random screening, ISM greedy, simulated annealing, genetic algorithm, AdaLead, vanilla categorical activation maximization, Fast SeqProp, CbAS, and DEN.
- Observed-data reference: the 100 highest-activity sequences in each weak model's training pool (`Train Top-100`). This reference uses known training labels, makes no weak-model queries, and is deterministic.
- Optimization budget: 10,000 weak-model evaluations per optimizer replicate.
- Output: 100 unique training-set-external sequences per optimizer replicate; 100 observed sequences for `Train Top-100`.
- Evaluation: exact TFBind8 exhaustive-landscape activity, queried only after optimization.

The experiment contains 600 model/method/replicate records and 60,000 evaluated output sequences.

## Performance figures

![Five-seed performance curves](../results/tfbind8_optimization_all_models_5seeds/multiseed_learning_curves.png)

![Mean activity and gain over random](../results/tfbind8_optimization_all_models_5seeds/multiseed_heatmaps.png)

### Four requested metrics

![Mean, median, top-1% hits, and best activity curves](../results/tfbind8_optimization_all_models_5seeds/four_metric_curves.png)

![Mean, median, top-1% hits, and best activity heatmaps](../results/tfbind8_optimization_all_models_5seeds/four_metric_heatmaps.png)

## Best mean-activity method for each weak model

| Training pool | Best method | Mean activity | SD across seeds | Gain over Random | Mean Top-1% hits |
|---:|---|---:|---:|---:|---:|
| 1% | Random | -0.02295 | 0.01296 | 0.00000 | 0.2 |
| 5% | Random | 0.05926 | 0.01253 | 0.00000 | 1.2 |
| 10% | Vanilla AM | 0.20099 | 0.01015 | +0.02434 | 2.8 |
| 20% | Fast SeqProp | 0.35196 | 0.00347 | +0.05073 | 28.8 |
| 30% | Simulated annealing | 0.36962 | 0.00064 | +0.04719 | 30.6 |
| 40% | Simulated annealing | 0.37397 | 0.00027 | +0.03239 | 34.2 |
| 50% | ISM greedy | 0.38859 | 0.00075 | +0.02705 | 43.6 |
| 60% | ISM greedy | 0.40066 | 0.00020 | +0.01675 | 52.2 |
| 70% | ISM greedy | 0.41174 | 0.00111 | +0.01854 | 56.4 |
| 80% | Simulated annealing | 0.42392 | 0.00038 | +0.00451 | 66.6 |
| 90% | ISM greedy | 0.43517 | 0.00050 | +0.00354 | 77.0 |
| 95% | ISM greedy | 0.44607 | 0.00024 | +0.00064 | 87.2 |

## Overall optimizer ranking

Methods are ranked within each of the 60 weak-model/seed conditions by the mean exact activity of their 100 designs. `Train Top-100` is excluded because it is an observed-data reference, not an optimizer.

| Rank | Optimizer | Mean rank | Median rank | Condition wins / 60 |
|---:|---|---:|---:|---:|
| 1 | ISM greedy | 2.483 | 2 | 22 |
| 2 | Simulated annealing | 2.650 | 2 | 19 |
| 3 | Fast SeqProp | 3.383 | 3 | 6 |
| 4 | Random screening | 4.333 | 4 | 3 |
| 5 | Genetic algorithm | 4.967 | 5 | 2 |
| 6 | Vanilla AM | 5.433 | 6 | 8 |
| 7 | AdaLead | 7.133 | 8 | 0 |
| 7 | CbAS | 7.133 | 8 | 0 |
| 9 | DEN | 7.483 | 8 | 0 |

## Interpretation

Random screening is the strongest method at 1% and 5%, indicating that those weak models do not yet provide a reliable local optimization landscape. Vanilla AM wins at 10%, and Fast SeqProp has the largest matched-random gain at 20%. From 30% onward, ISM greedy and simulated annealing are the most consistently effective methods.

The advantage over Random peaks around 20%-40% training coverage. At 80%-95%, the gap becomes small because excluding the known lowest-activity training pool leaves an increasingly enriched candidate space; under the 95% model, every valid novel candidate belongs to the true top 5% by construction.

The deterministic Train Top-100 curve is below designed sequences at every fraction. This is expected because each training pool contains the lowest-activity fraction of the landscape; the reference measures the best already observed sequence rather than the quality of novel optimization.

AdaLead found the exact global optimum (activity 0.49105) once under the 95% model, but its mean candidate activity remained below ISM and simulated annealing. Thus, best-single-sequence performance and stable batch quality lead to different conclusions.

## Outputs

- `summary.csv`: all 600 model/method/replicate summaries.
- `aggregate_summary.csv`: five-seed mean, SD, SEM, and matched-random gain.
- `all_candidates.csv`: all 60,000 evaluated output rows.
- `overall_algorithm_ranking.csv`: optimizer ranks across 60 conditions.
- Per-run candidate CSV and summary JSON files are stored under each model's `run_00` through `run_04` directory.
