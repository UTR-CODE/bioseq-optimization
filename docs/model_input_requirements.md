# Model input requirements

Black-box optimizers accept a scorer with `score(sequences)` returning one or
more predictions per sequence. Gradient optimizers also require
`score_soft(probabilities)`, preserving gradients from predictions to the
shared `[batch, length, A/C/G/T]` input.

| Models | Gradient input requirements |
|---|---|
| Optimus 5-Prime | Channels-first one-hot input |
| DeepSTARR and DeepSTARR-Mouse/Embryo | Channels-first one-hot input |
| Agarwal HepG2, K562, WTC11 | Channels-first input in A/G/C/T order |
| Cetnar stability | Channels-last input; use CPU for gradients in eval mode |
| DREAM LegNet | Model-specific context and channel adapter |
| PARM K562 | Adapter insertion and padding |
| UTR-CODE | Transcript and fixed-context adapter |
| CDS Stability | Codon tokenization and auxiliary-input adapter |

Use `DifferentiableSequencePredictor` with the model's input conversion
and a suitable `soft_input_transform`. Both paths must represent the same
sequence, including padding, channel order and fixed context.

Sequence length must match the scorer's accepted input. For transcript models,
configure the designed region and fixed flanking regions in the scorer.
Nucleotide optimization does not automatically preserve amino-acid sequences
or enforce an open reading frame.
