# Agarwal 2025 lentiMPRA

`MPRALegNet` is a faithful PyTorch port of the authors' public
[`human_legnet`](https://github.com/autosome-ru/human_legnet) model.  It is
the paper's strong sequence-only CNN and is trained independently for HepG2,
K562 and WTC11; it is not a three-output multi-task model.

- Input: complete 230-nt construct (15-nt adapter + 200-nt cCRE + 15-nt adapter).
- Encoding: A/G/C/T channel order plus the published fifth orientation channel.
- Architecture: 64-channel stem; four EfficientNetV2-like depthwise-convolution
  and squeeze-excitation blocks (80/96/112/128 channels); global average pool;
  scalar regression head.  Parameter count: 1,323,169.
- Training: random 0--21 nt vector-context shift, random reverse complement,
  AdamW, OneCycle learning rate, gradient clipping and ten-fold CV.

The three standardised full libraries are:

| Dataset | Rows |
| --- | ---: |
| `agarwal2025_lentimpra_hepg2` | 245,852 |
| `agarwal2025_lentimpra_k562` | 393,328 |
| `agarwal2025_lentimpra_wtc11` | 92,370 |

Smoke test (one outer CV split, deliberately not a reportable model):

```bash
python -m BioSeqBench.training.train_agarwal_mpralegnet \
  --mode demo --epochs 1 --max-train-batches 2 --max-eval-batches 2 --device cuda
```

The source-compatible demo is one model per cell type, with fold 1 held out as
test and fold 2 as validation.  BioSeqBench's practical strong-model protocol
uses one deterministic validation fold for each held-out test fold, so it
trains ten models per cell type (30 total):

```bash
python -m BioSeqBench.training.train_agarwal_mpralegnet --mode 10fold --device cuda
```

The paper's exhaustive source protocol is more expensive: for each cell type
it trains all 90 ordered validation/test fold pairs:

```bash
python -m BioSeqBench.training.train_agarwal_mpralegnet --mode cv --device cuda
```

For the exact paper-style three-cell run (author `rev=0` training rows,
`*_averaged.tsv` final labels, and nine-model ensemble per outer test fold),
use the resumable wrapper after any currently running lentiMPRA job has exited:

```bash
bash scripts/train_agarwal2025_mpralegnet_paper_cv.sh
```

The completed practical 10-fold runs are summarised with test-time
forward/reverse-complement averaging (TTA) by:

```bash
python -m BioSeqBench.evaluation.summarize_agarwal_mpralegnet
```

This reports the mean and sample standard deviation across outer folds rather
than a pooled correlation; individual OOF prediction vectors are not retained.
