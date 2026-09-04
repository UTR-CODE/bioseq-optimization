# Zhao 2022 Ptrc promoter models

Zhao et al. selected XGBoost as their best model for the 3,665 measured Ptrc
variants.  Their Figure 4C evaluates `log10(fluorescence/OD600)` and reports
R²=0.8846, MAE=0.15, and Pearson=0.94 on 127 designed promoters.  The paper
does not release its exact train/test identities, hyperparameters, or model
file.  `train_zhao2022_ptrc.py` is therefore a functional reproduction:
positional A/C/G/T one-hot features, validation-selected XGBoost, and the
benchmark's deterministic 80/10/10 split.

The same training script also includes `PtrcPromoterCNN`, a compact
non-pretrained, sequence-only differentiable comparator.  It uses the same
log10 target and never selects on the test split.

```bash
python -m BioSeqBench.training.train_zhao2022_ptrc --mode all --device cpu
```
