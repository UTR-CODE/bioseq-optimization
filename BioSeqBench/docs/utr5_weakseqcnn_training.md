# 5′UTR standardized weak-model training

## Final protocol

All 5′UTR weak predictors use the task-independent `WeakSeqCNN-v1` architecture
defined for the standardized nucleotide benchmark. The network has 194,882 trainable parameters. For
comparison, `Optimus5PrimePaper` has 474,681 parameters, so the standardized
weak model is 41.1% of the Optimus parameter count.

The 260,000 rows of `BioSeqBench/dataset/utr5_mpra/train.csv` are ranked by
measured mean ribosome load (MRL), with sequence used as the deterministic tie
breaker. The visible pools are the lowest 1%, 5%, 10%, 20%, ..., 90%, and 95%.
Each visible pool is deterministically divided into 90% fit and 10% validation
data. This split is fixed across the five model seeds at a threshold. Model
selection monitors only the in-pool validation MSE.

The highest 5% of `train.csv` (13,000 sequences) is one fixed OOD test set. It
is disjoint from every visible pool, including the bottom-95% pool, and is never
used for fitting or early stopping. The separate official `test.csv` is not used
in this weak-model experiment.

Training uses MSE, AdamW (learning rate 1e-3, weight decay 1e-4), cosine
annealing, at most 100 epochs, patience 10, and batch size 1,024. Every member
has a distinct recorded seed. RNGs are seeded, while forced deterministic CUDA
kernels are disabled and cuDNN benchmarking is enabled; this execution choice
is recorded in `protocol.json` and is identical for all 60 models.

## Ensemble results

Metrics below are computed from the mean prediction of the five members. `Val`
is the threshold-specific lower-tail validation split. `Top-5% test` is the
fixed high-activity tail from `train.csv`.

| Visible pool | Fit | Val | Val R² | Val Pearson | Val Spearman | Top-5% test R² | Test Pearson | Test Spearman | Test RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 2,340 | 260 | 0.001 | 0.106 | 0.042 | -549.463 | -0.020 | -0.018 | 5.981 |
| 5% | 11,700 | 1,300 | 0.031 | 0.190 | 0.211 | -435.841 | -0.046 | 0.025 | 5.328 |
| 10% | 23,400 | 2,600 | 0.225 | 0.476 | 0.488 | -355.700 | -0.062 | -0.001 | 4.815 |
| 20% | 46,800 | 5,200 | 0.526 | 0.726 | 0.702 | -314.130 | 0.064 | 0.083 | 4.526 |
| 30% | 70,200 | 7,800 | 0.693 | 0.832 | 0.831 | -303.867 | 0.090 | 0.080 | 4.451 |
| 40% | 93,600 | 10,400 | 0.784 | 0.885 | 0.890 | -234.736 | 0.099 | 0.055 | 3.914 |
| 50% | 117,000 | 13,000 | 0.845 | 0.920 | 0.920 | -120.983 | 0.023 | -0.014 | 2.816 |
| 60% | 140,400 | 15,600 | 0.871 | 0.933 | 0.922 | -49.437 | -0.083 | -0.058 | 1.811 |
| 70% | 163,800 | 18,200 | 0.890 | 0.943 | 0.921 | -31.133 | -0.167 | -0.091 | 1.445 |
| 80% | 187,200 | 20,800 | 0.899 | 0.948 | 0.920 | -17.033 | -0.277 | -0.146 | 1.083 |
| 90% | 210,600 | 23,400 | 0.913 | 0.955 | 0.926 | -11.080 | -0.297 | -0.091 | 0.886 |
| 95% | 222,300 | 24,700 | 0.915 | 0.957 | 0.927 | -8.542 | -0.282 | 0.021 | 0.788 |

The negative top-tail R² values are not a file or split error. The top-5% test
labels occupy a narrow, shifted range (mean 8.479, standard deviation 0.255),
whereas even the 95% ensemble remains systematically low (prediction mean
7.808 and maximum 8.036, below the test minimum 8.276). This is direct evidence
of lower-tail surrogate extrapolation error and should be retained when
benchmarking sequence optimizers.

## Artifacts and validation

- `results/utr5_weak_cnns/protocol.json`: complete data and training protocol.
- `results/utr5_weak_cnns/results.csv`: 12 ensemble-level rows.
- `results/utr5_weak_cnns/surrogate_results.csv`: 60 member-level rows.
- `results/utr5_weak_cnns/bottom_*pct/`: five checkpoints plus validation/test
  predictions and member metadata for each threshold.

The final audit found 60/60 compatible checkpoints, 12/12 threshold summaries,
five unique seeds per threshold, one shared top-5% test hash, and zero overlap
between the bottom-95% pool and the test set. All numeric result fields are
finite, checkpoint-backed ensemble inference succeeds, and the repository test
suite passes 39 tests.
