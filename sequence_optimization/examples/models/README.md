# Collected model examples

The optimizer package is dataset-independent. The examples in this directory
show how models collected in the BioSeqBench project can be wrapped as scorers:

- `WeakSeqCNNExample`: standardized CNN used for 5′UTR and other nucleotide tasks;
- `Optimus5PrimeExample`: Optimus 5′UTR model;
- `DeepSTARRExample`: two-task developmental/housekeeping enhancer model;
- `TFBind8Example`: TFBind8 CNN;
- `DreamLegNetExample`: DREAM Autosome/LegNet model.

These wrappers are intentionally optional. The core package (`api.py` and
`algorithms.py`) never imports datasets, checkpoints, or BioSeqBench model
implementations. For a new model, implement `score(sequences)` directly or use
`TorchSequenceScorer(model, encoder)`.

`catalog.py` lists the other collected DNA models as lazy import metadata. This
keeps the implementations in their original model package instead of creating
two divergent copies. A deployment can replace an entry with a checkpoint-aware
wrapper without changing any optimizer code.

`Optimus5PrimeExample.from_checkpoint()` and
`DeepSTARRExample.from_checkpoint()` use the repository's current benchmark
checkpoints by default. Both also accept an explicit checkpoint path.
