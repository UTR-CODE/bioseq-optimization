# Motif discovery

[Home](../README.md) · [Core API](core-api.md)

The second core function converts generated sequences into a motif catalog:

```python
from sequence_optimization import call_motifs
motifs = call_motifs(designs, model=scorer, output_dir="outputs/motifs")
```

Use the same scorer as in `generate_designs`. The optimization objective stored with the designs defines the attribution target for single-task, multi-task, and combined-model predictions.

Before calculating contributions, `call_motifs` checks that STREME and FIMO are executable. Missing tools produce an installation message immediately:

```bash
conda install -c conda-forge -c bioconda meme
```

Supply `streme="/path/to/streme"` and `fimo="/path/to/fimo"` when the executables are outside `PATH`. Reusing an existing catalog with `catalog=...` skips this discovery-tool check.

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

The output directory contains `contributions.npz`, `seqlets.csv/fa`, `background.fa`, `support.csv`, and **`catalog.meme`**. The MEME file contains the retained DNA PWMs and is the portable input to candidate selection.

Pass the exported file directly to selection:

```python
from sequence_optimization import select_candidates
selected = select_candidates(
    designs, motifs="outputs/motifs/catalog.meme", top_n=10,
    output_dir="outputs/panel_from_meme",
)
```

You can skip discovery entirely and replace that path with `"my_motifs.meme"`, produced by your own motif-calling method. Supply DNA letter-probability matrices with `ALPHABET= ACGT` and unique motif IDs. The selector performs FIMO scanning and motif-diversity selection without needing the prediction model or attribution scores.

To load an existing catalog as an in-memory `MotifCatalog` instead:

```python
motifs = call_motifs(designs, catalog="catalog.meme")
```

For catalog-level organization and annotation, [Almeida motif-clustering](https://github.com/bernardo-de-almeida/motif-clustering) and Tomtom can operate on the exported PWMs. The subsequent selection stage retains informative flank differences through continuous PWM similarity.
