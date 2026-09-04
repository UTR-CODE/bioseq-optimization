# Kosuri composability CNN

Kosuri et al. (2013) used interpretable compositional models rather than a deep neural network. BioSeqBench's canonical surrogate is a non-pretrained two-output CNN with separate promoter and RBS encoders, followed by a joint regression head. It predicts `log10` protein expression and RNA abundance, restoring raw units for evaluation. The split remains grouped by original promoter--RBS construct.

The canonical run uses 9,372/1,152/1,172 train/validation/test constructs and early-stops at epoch 136. On the held-out test set, its log10-scale Pearson/Spearman/R² are 0.955/0.949/0.912 for protein expression and 0.985/0.987/0.970 for RNA abundance. The saved checkpoint and per-construct predictions are under `BioSeqBench/checkpoints/strong_kosuri2013_v1/kosuri2013_composability/two_branch_cnn/`.

```bash
python -m BioSeqBench.training.train_kosuri2013_composability --device cuda
```
