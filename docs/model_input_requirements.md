# Model input requirements

`SequencePredictor` supplies both `score(sequences)` and
`score_soft(probabilities)`. Its default model input is channels-first A/C/G/T
with shape `[batch, 4, length]`.

| Models | Gradient input requirements |
|---|---|
| Optimus 5-Prime | Channels-first one-hot input |
| DeepSTARR and DeepSTARR-Mouse/Embryo | Channels-first one-hot input |
| Malinois | Fixed left/right assay context and channels-first input |
| Agarwal HepG2, K562, WTC11 | Channels-first input in A/G/C/T order |
| Cetnar stability | Channels-last input; use CPU for gradients in eval mode |
| DREAM LegNet | Model-specific context and channel adapter |
| PARM K562 | Adapter insertion and padding |
| UTR-CODE | Transcript and fixed-context adapter |
| CDS Stability | Codon tokenization and auxiliary-input adapter |

For another layout, channel order, or fixed context, subclass
`SequencePredictor` and override `prepare_input` with differentiable PyTorch
operations. See [custom PyTorch models](custom_models.md).

Sequence length must match the scorer's accepted input. For transcript models,
configure the designed region and fixed flanking regions in the scorer.
For CDS design, use [synonymous codon constraints](tasks/synonymous_cds.md)
to preserve the reference protein. Unconstrained nucleotide optimization does
not preserve amino-acid sequences or enforce an open reading frame.
