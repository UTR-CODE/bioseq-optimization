# Human 5′UTR MPRA model

## Standardized weak surrogate

Optimization weak models use the task-independent `WeakSeqCNN-v1` defined in
`BioSeqBench/models/weakseqcnn.py`, not the Optimus architecture. Every lower-tail
threshold uses the identical 194,882-parameter network: 64 channels, four
dilated residual blocks with dilations 1/2/4/8, masked mean/max/attention pooling,
and a scalar regression head. The 260,000 rows in `train.csv` are ranked by true
activity. Each lower-tail threshold is split deterministically into 90% fit and
10% in-pool validation data; this validation split is shared by all five model
seeds at that threshold. The highest-activity 5% of `train.csv` (13,000 rows) is
the fixed, disjoint OOD test set for every threshold. The official `test.csv` is
not used in this weak-surrogate experiment. The UTR training batch size is 1,024.

## Strong Optimus model

`Optimus5PrimePaper` is a sequence-only PyTorch reproduction of the authors' public
notebook: 50-nt one-hot input; three same-padded ReLU Conv1D layers (120 filters,
width 8); flatten; 40-unit ReLU dense layer; 0.2 dropout; linear MRL output. It uses
Adam (1e-3), MSE, batch size 128 and three epochs. The training target is standardized
using the 260,000-member training set only.

The data split is also the publication split: rank the `egfp_unmod_1` library by
`total_reads`, retain the top 280,000, use ranks 1--20,000 for test, and ranks
20,001--280,000 for train. The original experiment uses no validation set.
Reference: Sample et al., Nature Biotechnology 2019, doi:10.1038/s41587-019-0164-5;
[authors' notebook](https://github.com/pjsample/human_5utr_modeling/blob/master/modeling/training_MRL_CNN.ipynb).
