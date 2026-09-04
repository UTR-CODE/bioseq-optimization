# DeepSTARR model

`DeepSTARRPaper` is a layer-order-faithful PyTorch reproduction of the
Basset-derived model in de Almeida et al., *Nature Genetics* (2022),
doi:10.1038/s41588-022-01048-5. The author implementation is available in the
[DeepSTARR training notebook](https://github.com/bernardo-de-almeida/DeepSTARR/blob/main/DeepSTARR/DeepSTARR_training.ipynb).

Architecture: four `Conv1D -> BatchNorm -> ReLU -> MaxPool(2)` blocks with
filters `256, 60, 60, 120` and kernel widths `7, 3, 5, 3`, followed by two
`Dense(256) -> BatchNorm -> ReLU -> Dropout(0.4)` blocks and two independent
one-unit heads for developmental and housekeeping activity. The implementation
also uses Keras-compatible Glorot initialisation and BatchNorm running-statistic
momentum (`0.01` in PyTorch equals Keras `momentum=0.99`).

Reproduce the reference run from the repository root:

```bash
/home/hexin/miniconda3/envs/UTR/bin/python -u -m BioSeqBench.training.train_deepstarr_paper --device cuda --num-workers 8
```

The source Zenodo files contain each sequence and its reverse complement. The
script reconstructs this augmentation for training and validation; it reports
both the source-protocol test score (41,186 rows) and the benchmark score on
the 20,593 unique forward sequences. With seed 1234, early stopping selected
epoch 5 (training stopped at epoch 15):

| Test protocol | developmental Pearson | housekeeping Pearson |
| --- | ---: | ---: |
| Paper | 0.6800 | 0.7400 |
| PyTorch reproduction | 0.6668 | 0.7485 |
| Unique-sequence benchmark | 0.6647 | 0.7486 |

Saved artefacts: `BioSeqBench/checkpoints/strong_deepstarr_paper_v1/deepstarr/deepstarr_paper/` and `BioSeqBench/results/deepstarr_paper_reproduction.csv`.
