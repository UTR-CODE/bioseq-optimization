# TFBind8 weak-model optimization benchmark

## Protocol

- Weak models: CNN ensembles trained on the lowest 1%, 50%, and 95% of the TFBind8 landscape.
- Optimizers: random screening, ISM greedy/SSWM, simulated annealing, genetic algorithm, AdaLead, vanilla categorical activation maximization, Fast SeqProp, CbAS, and DEN.
- Budget: 10,000 weak-model evaluations per weak-model/optimizer combination.
- Output: exactly 100 unique sequences outside that weak model's visible training pool.
- Initialization: 50 high-activity visible training sequences plus 50 random sequences; methods that do not require seeds ignore this pool.
- Evaluation: the exact TFBind8 exhaustive landscape is consulted only after candidate generation.
- Seed: 20260825, with a deterministic offset for each combination.

The global TFBind8 maximum activity is 0.49105. No method found that exact sequence in this run.

## Performance figures

![Candidate activity distributions](../results/tfbind8_optimization_benchmark/activity_distributions.png)

![Performance heatmaps](../results/tfbind8_optimization_benchmark/performance_heatmaps.png)

## Exact activity results

| Weak model | Optimizer | Mean of 100 | Best | Mean of top 10 | Top-1% hits / 100 |
|---:|---|---:|---:|---:|---:|
| 1% | Random screening | -0.028164 | 0.40867 | 0.314378 | 0 |
| 1% | ISM greedy | -0.040108 | 0.31233 | 0.260291 | 0 |
| 1% | Simulated annealing | -0.030344 | 0.31233 | 0.269606 | 0 |
| 1% | Genetic algorithm | **-0.021060** | 0.31233 | 0.284585 | 0 |
| 1% | AdaLead | -0.034312 | 0.31233 | 0.269606 | 0 |
| 1% | Vanilla activation maximization | -0.034070 | 0.31233 | 0.283174 | 0 |
| 1% | Fast SeqProp | -0.022673 | 0.31233 | 0.269606 | 0 |
| 1% | CbAS | -0.086795 | 0.30919 | 0.184181 | 0 |
| 1% | DEN | -0.034312 | 0.31233 | 0.269606 | 0 |
| 50% | Random screening | 0.363448 | **0.47427** | 0.464040 | 32 |
| 50% | ISM greedy | 0.388502 | 0.47115 | **0.466542** | **44** |
| 50% | Simulated annealing | **0.388776** | 0.47115 | **0.466542** | 43 |
| 50% | Genetic algorithm | 0.379969 | 0.47115 | 0.461585 | 37 |
| 50% | AdaLead | 0.264740 | 0.46490 | 0.449793 | 18 |
| 50% | Vanilla activation maximization | 0.340065 | 0.46796 | 0.453278 | 18 |
| 50% | Fast SeqProp | 0.376758 | 0.47115 | 0.463871 | 36 |
| 50% | CbAS | 0.343637 | 0.45964 | 0.449244 | 21 |
| 50% | DEN | 0.332614 | 0.46796 | 0.439443 | 12 |
| 95% | Random screening | 0.444914 | **0.49088** | **0.484188** | 87 |
| 95% | ISM greedy | **0.446412** | **0.49088** | **0.484188** | **89** |
| 95% | Simulated annealing | 0.446034 | **0.49088** | **0.484188** | 87 |
| 95% | Genetic algorithm | 0.429437 | **0.49088** | 0.479390 | 68 |
| 95% | AdaLead | 0.400679 | **0.49088** | 0.480409 | 45 |
| 95% | Vanilla activation maximization | 0.436038 | 0.48850 | 0.480512 | 78 |
| 95% | Fast SeqProp | 0.443063 | **0.49088** | 0.483742 | 86 |
| 95% | CbAS | 0.435793 | **0.49088** | 0.479333 | 78 |
| 95% | DEN | 0.372029 | 0.47847 | 0.455878 | 22 |

## Interpretation

The 1% model is too weak for reliable optimization: none of the 900 proposed sequences reached the landscape's top 1%. Random screening found the best individual sequence, while the genetic algorithm had the highest (but still poor) mean activity.

At 50%, simulated annealing had the highest mean activity, while ISM greedy produced the most top-1% sequences. At 95%, ISM greedy was narrowly best by both mean activity and top-1% hit count; simulated annealing and random screening were close.

The 95% condition needs careful interpretation. Because the weak model sees the lowest 95% and candidates are required to be outside its training pool, every valid candidate is already in the landscape's true top 5%. Consequently, absolute activity is inflated even for random screening. Rankings should therefore be interpreted as enrichment over the matched random baseline, not only by raw activity. The current table is one deterministic run; multiple seeds are needed for uncertainty intervals and a final algorithm ranking.

## Reproduction and outputs

Run:

```bash
/home/hexin/miniconda3/envs/UTR/bin/python -u \
  -m BioSeqBench.optimization.run_tfbind8_benchmark \
  --output-dir BioSeqBench/results/tfbind8_optimization_benchmark
```

The output directory contains `protocol.json`, `summary.csv`, `all_candidates.csv`, and one candidate CSV plus summary JSON for each combination.
