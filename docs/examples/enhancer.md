# Optimize enhancer specificity with Malinois

[Notebook](../../examples/enhancer.ipynb) · [Motif-discovery helper](../../examples/fig2_motif_catalog.py) · [Installation](../installation.md)

The notebook reproduces the Fig. 2 enhancer workflow. Clone the repository and run the notebook in a fresh GPU kernel with MEME Suite on PATH. Keep the notebook and its helper in the same directory.

1. Generate 100 candidates per optimizer using Malinois K562 activity minus the higher of the HepG2 and SK-N-SH predictions. The run uses base seed 42, FP32, fixed prediction batches and 90% gain.
2. Discover motifs from all seqlets, then repeat discovery with one representative per seqlet sequence or reverse complement. Scan the full pool with FIMO and retain motifs that pass site support and spatial redundancy filtering.
3. Select ten sequences from the highest-scoring 80% of unique candidates using the combined catalog.

The saved run contains 1,100 optimizer returns, 1,084 unique enhancers and 13 final motifs.
