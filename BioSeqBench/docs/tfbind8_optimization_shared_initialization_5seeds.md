# TFBind8 five-seed benchmark with shared initialization

## Revised initialization protocol

For every weak-model/replicate condition, one shared set of 100 unique seeds is created:

- 50 highest-activity sequences in the visible lower-tail training pool.
- 50 sequences sampled uniformly from the complete 65,536-sequence TFBind8 space.

The random draw is not conditioned on training membership. Only the 50 selected Train-Top sequences are excluded from the random sampling pool to prevent duplicates. ISM, simulated annealing, GA, AdaLead, vanilla activation maximization, Fast SeqProp, CbAS, and DEN use exactly the same initialization ID within a condition. Random screening and Train Top-100 do not use this seed set.

CbAS fits its initial autoregressive distribution to the shared seeds. DEN first trains its generator to reconstruct the shared seeds and achieved 100% nucleotide reconstruction accuracy in all 60 conditions. GA uses the 100 shared seeds as its complete initial population.

AdaLead reserves an activity-fraction-dependent portion of its 10,000-query budget for uniform full-space proposals. These proposals are not conditioned on training membership and ensure that 100 unique novel outputs remain feasible for large training pools.

## Initialization audit

- 60 shared seed sets: 12 weak models × 5 replicates.
- Every set contains exactly 50 `train_top` and 50 `full_space_random` sequences.
- Every set contains 100 unique sequences.
- All eight seeded optimizers have the same initialization ID within a condition.
- Full-space random seeds found inside the training pool: 2.0% at the 1% model, 48.4% at the 50% model, 90.0% at the 90% model, and 91.6% at the 95% model. This confirms that random seeds were not restricted to unseen sequences.

## Results

![Four performance curves](../results/tfbind8_optimization_all_models_5seeds_shared_init/four_metric_curves.png)

![Four performance heatmaps](../results/tfbind8_optimization_all_models_5seeds_shared_init/four_metric_heatmaps.png)

| Training pool | Best mean-activity method | Mean activity | SD | Mean Top-1% hits | Gain over Random |
|---:|---|---:|---:|---:|---:|
| 1% | Random | -0.02295 | 0.01296 | 0.2 | 0.00000 |
| 5% | Random | 0.05926 | 0.01253 | 1.2 | 0.00000 |
| 10% | CbAS | 0.20665 | 0.00563 | 6.4 | +0.03000 |
| 20% | ISM | 0.35156 | 0.00348 | 28.8 | +0.05033 |
| 30% | ISM | 0.37129 | 0.00112 | 30.8 | +0.04886 |
| 40% | CbAS | 0.37466 | 0.00145 | 35.4 | +0.03308 |
| 50% | CbAS | 0.38975 | 0.00115 | 43.6 | +0.02821 |
| 60% | ISM | 0.40067 | 0.00055 | 52.4 | +0.01676 |
| 70% | ISM | 0.41078 | 0.00178 | 56.2 | +0.01757 |
| 80% | CbAS | 0.42403 | 0.00052 | 66.6 | +0.00462 |
| 90% | ISM | 0.43505 | 0.00054 | 77.0 | +0.00342 |
| 95% | ISM | 0.44615 | 0.00041 | 88.0 | +0.00073 |

## Overall rank across 60 conditions

| Rank | Optimizer | Mean rank | Wins / 60 |
|---:|---|---:|---:|
| 1 | ISM | 3.083 | 20 |
| 2 | Simulated annealing | 3.092 | 11 |
| 3 | CbAS | 3.167 | 15 |
| 4 | Fast SeqProp | 4.083 | 3 |
| 5 | Random screening | 4.900 | 3 |
| 6 | Vanilla AM | 5.767 | 4 |
| 7 | Genetic algorithm | 5.817 | 1 |
| 8 | AdaLead | 7.325 | 0 |
| 9 | DEN | 7.767 | 2 |

The largest change relative to the previous initialization protocol is CbAS: its mean activity averaged across weak models increased by 0.0681. ISM and simulated annealing remain the most consistently high-ranking methods. Genetic algorithm and DEN became weaker on average after removal of their additional method-specific random initialization.

The exact global optimum (0.49105) was found four times under the 95% weak model: twice by GA and twice by AdaLead. This does not change the batch-quality conclusion because ISM, simulated annealing, and CbAS have higher mean and median activity.
