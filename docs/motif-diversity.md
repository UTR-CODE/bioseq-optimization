# Candidate selection: sequence or motif diversity

[Home](../README.md) · [Core API](core-api.md)

The third core function selects a high-quality diverse panel. Its distance representation is determined by the `motifs` argument.

## 1. Default Hamming selection

```python
from MotifSift import select_candidates
selected = select_candidates(designs, top_n=10, min_score=gate)
```

Without `motifs`, selection uses Hamming Max-Min: start with the highest-scoring design, then maximize minimum nucleotide distance to the selected panel. Near distance ties favor higher scores.

## 2. Motif-based selection

```python
selected = select_candidates(
    designs, top_n=10, motifs=motifs, min_score=gate,
    output_dir="outputs/motif_panel",
)
```

With motifs, the function scans and filters motif occurrences, builds the composition and positional representations, and performs hierarchical Max-Min selection.

## 3. Motif composition

A binary Sequence × Motif matrix records whether each motif occurs. PWM similarities are the best strand- and shift-aligned cosines of information-weighted contrasts, retaining informative flanks.

For RNA, create the catalog with `call_motifs(..., reverse_complement=False)` or reload it with `load_motifs()`. To use a MEME file for RNA, first wrap it with `call_motifs(designs, catalog=path, reverse_complement=False)`. Selection uses the catalog’s strand setting for PWM alignment and FIMO scanning.

Presence propagates through the motif similarity matrix:

\[
F_{ik}=\max_{m\in M_i} S_{mk}.
\]

Composition distance is the Soft-Jaccard distance:

\[
D_C(i,j)=1-\frac{\sum_k\min(F_{ik},F_{jk})}{\sum_k\max(F_{ik},F_{jk})}.
\]

## 4. Motif positions

Every occurrence is represented by its motif identity and center coordinate. Hungarian matching first maximizes the number of compatible motif pairs, then minimizes total positional displacement. The reported distance is mean displacement over matched occurrences. Repeated instances are retained; unmatched instances are omitted.

FIMO defaults are `p ≤ 0.001` and `q ≤ 0.1`. Same-motif overlapping footprints are consolidated, and motifs require more than 30 physical sites on unique DNA. Motifs are removed when more than 80% of either site set is covered by the other; representatives prioritize site count, PWM information and motif ID. See [candidate-wide filtering](motif-discovery.md#candidate-wide-site-filtering).

## 5. Hierarchical selection and outputs

Selection starts with the highest-scoring eligible design, then prioritizes:

1. Maximum minimum composition distance to the selected panel.
2. Maximum minimum matched positional distance among composition near-ties.
3. Higher prediction score among positional near-ties.

```python
selected = select_candidates(
    designs, motifs=motifs, top_n=10, min_score=gate,
)
print(selected.attrs["selection_method"])  # motif
```

The returned table has the same sequence, score, and provenance columns as Hamming selection. An output directory additionally stores motif sites, presence matrices, PWM similarity, composition and position distances, match counts, CSV/FASTA, and the selection trace.

See the [reference-data notebook](../examples/motif_diversity.ipynb) for selection from a saved catalog and hits.

## When a panel cannot be selected

If discovery yields no supported motifs, omit `motifs` to select by Hamming distance. If fewer than `top_n` unique sequences meet the score cutoff, reduce `top_n` or adjust `min_score` to reflect the experiment’s requirements. Motif-based selection also needs sufficient motif evidence in the eligible pool.

## Outputs

`selected_sequences.csv/fa` contain the panel. `motif_sites.csv`, `presence_matrix.csv`, `motif_similarity.csv`, `distance_matrices.npz`, and `selection_trace.json` describe the motif-based selection. Hamming mode exports `hamming_distances.csv` and the same sequence and trace files.
