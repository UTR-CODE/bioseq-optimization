# Motif discovery

[Home](../README.md) · [Core API](core-api.md)

The second core function converts generated sequences into a motif catalog:

```python
from sequence_optimization import call_motifs
motifs = call_motifs(designs, model=scorer, output_dir="outputs/motifs")
```

Use the same scorer as in `generate_designs`. The optimization objective stored with the designs defines the attribution target for single-task, multi-task, and combined-model predictions.

## Workflow

```text
Unique designs → ISM contributions → positive seqlets
               → STREME against shuffled controls → supported PWMs
```

ISM scores the three alternative bases at each position. The projected observed-base contribution is the original score minus the four-base mean prediction. Positive fragments are extracted with tangermeme's recursive seqlet caller.

The discovery stage compares seqlet sequences against length- and dinucleotide-matched shuffled controls. It retains motifs supported by at least ten seqlets from at least three parent DNA sequences.

## Discovery parameters

```python
motifs = call_motifs(
    designs, model=scorer,
    output_dir="outputs/motifs_custom",
    seqlet_threshold=0.01, min_width=6, max_width=15,
    max_seqlet_len=25, flanks=2, max_motifs=30,
    min_seqlets=10, min_parents=3, seed=42,
)
```

`max_motifs` controls the maximum number of discovery candidates. The final count follows the data and support filters.

## Existing contributions

Projected ISM or DeepLIFTShap contributions can be supplied directly:

```python
motifs = call_motifs(designs, contributions=contributions, output_dir="outputs/motifs_from_scores")
```

For an array, rows correspond to the input design records and columns to nucleotide positions. A DataFrame indexed by `sequence_id` provides explicit alignment. Exact sequence duplicates are collapsed internally.

## Inspect and reuse the catalog

```python
print(len(motifs))
print(motifs.support)
print(motifs.seqlets[["seqlet_id", "sequence_id", "start", "end", "dna"]].head())
```

The output directory contains `contributions.npz`, `seqlets.csv/fa`, `background.fa`, `support.csv`, and `catalog.meme`.

An existing MEME catalog can be reused through the same interface:

```python
motifs = call_motifs(designs, catalog="catalog.meme")
```

For catalog-level organization and annotation, [Almeida motif-clustering](https://github.com/bernardo-de-almeida/motif-clustering) and Tomtom can operate on the exported PWMs. The subsequent selection stage retains informative flank differences through continuous PWM similarity.
