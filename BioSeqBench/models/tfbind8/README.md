# TFBind8 model

`TFBind8DeepBind` is a PyTorch implementation of the DeepBind regression
`maxpool/M2` template in the authors' supplementary source code:

`Conv1D(16 motif filters) → bias → ReLU → global max-pool → dropout(0.25) → Dense(32) → ReLU → linear regression`.

The released DeepBind configuration scans 16-nt motifs because its PBM probes are
longer. TFBind8 contains complete 8-mers only, so this task-specific adaptation
uses an 8-nt valid motif scanner; pooling consequently has one spatial position.
The model averages forward and reverse-complement predictions, matching the
original TF pipeline's reverse-complement augmentation. The published TFBind8
labels are exactly RC invariant, and canonical RC pairs are grouped into the
same train/validation/test split to avoid leakage.

Train with the paper-matched optimization family:

```bash
python -m BioSeqBench.training.train --dataset tfbind8 --model deepbind \
  --epochs 500 --batch-size 64 --optimizer sgd --momentum 0.95 \
  --learning-rate 0.01 --weight-decay 1e-5 --standardize-targets --patience 50
```

Reference: Alipanahi et al., *Nature Biotechnology* 2015,
doi:10.1038/nbt.3300; `cfg/regression/maxpool/M2.cfg` in supplementary archive
`41587_2015_BFnbt3300_MOESM62_ESM.zip`.
