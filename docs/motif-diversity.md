# Motif composition and positional diversity

[Home](../README.md) · [Motif discovery](motif-discovery.md)

Select a panel that combines high predicted activity with diverse motif composition and positions.

## 1. Scan motifs onto the candidate sequences

```python
from sequence_optimization.motifs import read_meme, scan_fimo
catalog = read_meme("outputs/motifs/catalog.meme")
hits = scan_fimo(
    "outputs/motifs/catalog.meme",
    "outputs/enhancer/unique_candidates.fa",
    "outputs/enhancer_fimo",
    p_threshold=0.001,
)
```

The workflow uses FIMO hits with `p ≤ 0.001` and `q ≤ 0.05`. For the same motif and sequence, overlapping footprints covering at least 70% of the shorter hit are represented by the best-p-value hit. Motifs require at least 30 physical sites across the 1,100 source records. Exact-DNA multiplicity is retained for this record-level support count.

## 2. Motif similarity

```python
from sequence_optimization.motif_diversity import pwm_similarity_matrix
motif_ids = list(catalog)
similarity, alignments = pwm_similarity_matrix([catalog[m] for m in motif_ids], min_overlap=6)
```

The similarity is the maximum strand- and shift-aligned cosine between information-weighted PWM contrasts. Full informative flanks contribute to the normalization.

## 3. Composition distance

Construct a binary Sequence × Motif presence matrix. Propagate motif presence through PWM similarity:

\[
F_{ik}=\max_{m\in M_i} S_{mk}.
\]

The Soft-Jaccard composition distance is

\[
D_C(i,j)=1-\frac{\sum_k\min(F_{ik},F_{jk})}{\sum_k\max(F_{ik},F_{jk})}.
\]

With an identity motif similarity matrix, this becomes binary Jaccard distance. Motif counts and positions enter the next stage separately.

## 4. Positional distance

Represent every motif occurrence by its motif ID and center coordinate. Match identical or similar motifs with Hungarian assignment, allowing multiple occurrences of each motif. Matching first maximizes the number of compatible pairs, then minimizes total absolute positional displacement.

The distance is the mean displacement of matched sites in base pairs. Unmatched sites are omitted. Match counts accompany the distance matrix and define which pairs contribute positional evidence.

## 5. Hierarchical Max-Min selection

```python
import pandas as pd
from sequence_optimization.motifs import select_motif_panel
candidates = pd.read_csv("outputs/enhancer/all_candidates.csv")
selected, trace = select_motif_panel(
    candidates, hits, catalog,
    output_dir="outputs/enhancer_top10",
    top_n=10, min_score=20., min_sites=30,
    match_similarity=0.5, composition_tie=0.02, position_tie_bp=2.0,
)
```

Set `min_score` in the predictor's activity units. Selection proceeds as follows:

1. Start with the highest-scoring eligible sequence.
2. Maximize minimum composition distance to the selected panel.
3. Among composition distances within 0.02 of the best value, maximize minimum matched positional distance.
4. Among positional distances within 2 bp of the best value, select the highest score.

Sequences carrying catalog motifs form the eligible motif-supported pool. Pairwise positional comparisons use matched motifs; a candidate with no comparable selected sequence receives a positional value of zero.

## 6. Reference-data example

The bundled K562 data contain 1,100 candidate records, 1,084 unique sequences, 11 motif models, and 12,164 physical sites. This example reconstructs the ordered Top10 directly from the saved predictions and motif hits.

```python
from pathlib import Path
import pandas as pd
from sequence_optimization.motifs import read_meme, select_motif_panel

DATA = Path("examples/data")
candidates = pd.read_csv(DATA / "k562_candidates_1100.csv")
hits = pd.read_csv(DATA / "k562_fimo_sites.csv")
catalog = read_meme(DATA / "k562_catalog.meme")
selected, trace = select_motif_panel(
    candidates, hits, catalog,
    output_dir="outputs/reference_top10",
    top_n=10, min_score=20., min_sites=30,
)
print(selected[["selection_rank", "sequence_id", "score", "present_motifs"]])
```

The [reference notebook](../examples/motif_diversity.ipynb) runs this selection and compares it with the saved Top10.

## Outputs

| File | Contents |
|---|---|
| `selected_sequences.csv/fa` | Panel sequences, scores, motifs, algorithm provenance |
| `presence_matrix.csv` | Unique-sequence motif presence |
| `all_record_presence_matrix.csv` | Motif presence for every source record |
| `motif_sites.csv` | Motif identities, strands, and coordinates |
| `motif_similarity.csv` | PWM similarity matrix |
| `distance_matrices.npz` | Composition distances, positional distances, and match counts |
| `selection_trace.json` | Sequence chosen at every Max-Min step |

Distance matrices are indexed by unique sequence IDs. Source-record provenance maps these matrices back to the complete 1,100-record pool.
