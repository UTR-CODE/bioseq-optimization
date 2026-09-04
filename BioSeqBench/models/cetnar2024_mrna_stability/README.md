# Cetnar 2024 mRNA stability

Target: `k_fit` in min^-1; smaller values correspond to more stable RNA.
`half_life_min` is retained as metadata, not a second training target.

## Paper reproduction: hybrid biophysical LightGBM

`paper_lightgbm.py` faithfully follows the public `ModelTrainingTesting.py`:
five LightGBM timepoint models predict log mRNA level at T0/T2/T4/T8/T16,
then a sixth LightGBM model predicts `k_fit` using those five predictions plus
the five-isoform biophysical feature set. The input has 460 timepoint features
and 465 decay-model features (RppH, transcription/translation rates, RNA
structure, ssRNA composition, G-quadruplex/i-motif/CsrA indicators, and 80
categorical nucleotide positions for each isoform). It is not a deep model.

```bash
python -m BioSeqBench.training.train_cetnar_paper_lightgbm
```

The run uses the authors' fixed raw membership (36,219 train / 9,064 test) and
the script's published tree hyperparameters. With LightGBM 4.7.0, test R2 is
0.4254 versus 0.43 in the paper. The five timepoint test R2 values are
0.6861, 0.6462, 0.6044, 0.5231 and 0.4279, matching the paper values
0.69, 0.65, 0.60, 0.53 and 0.43.

## Sequence-only deep-learning baseline

`CetnarStabilityCNNBiLSTM` is a non-pretrained PyTorch model: two motif CNN
layers (96 channels; kernel sizes 7 and 5), a 64-dimensional bidirectional
LSTM, masked mean pooling and a 64-unit regression head. It consumes only the
RNA sequence, not calculated biophysical features. A deterministic 10%
sequence-grouped validation split is formed inside the authors' training data;
the author test split is untouched for model selection.

```bash
python -m BioSeqBench.training.train_cetnar_sequence_dl --device cuda --num-workers 4
```

On the 9,042 positive-`k_fit` canonical test observations, this model reaches
Pearson 0.7406, Spearman 0.7106 and R2 0.5479 (best epoch 11). There are 21
test observations whose exact RNA sequence also occurs in the author training
membership. Excluding them gives the strict sequence-unseen score: Pearson
0.7404, Spearman 0.7104 and R2 0.5477.

Results and checkpoints are stored under
`BioSeqBench/checkpoints/strong_cetnar_paper_v1/` and
`BioSeqBench/checkpoints/strong_cetnar_sequence_v1/`.
