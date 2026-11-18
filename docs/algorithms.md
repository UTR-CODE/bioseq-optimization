# Optimizers

MotifSift runs eleven optimizers over a common starting pool: AdaLead, BaseSearch, beam search, cross-entropy, DEN, Fast SeqProp, genetic algorithm, in silico mutagenesis, Ledidi, simulated annealing, and vanilla activation maximization.

Each contributes up to `num_candidates` sequences per requested gain stage. To run a subset, pass the optimizer names in `algorithms`.
