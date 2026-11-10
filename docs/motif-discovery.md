# Motif discovery from attribution seqlets

[Home](../README.md) · [Core API](core-api.md) · [Method comparison](motif-discovery-comparison.md)

The second core function converts designed sequences and model contributions into a motif catalog:

```python
from sequence_optimization import call_motifs

motifs = call_motifs(
    designs, model=scorer, output_dir="outputs/motifs",
    min_seqlets=10, min_parents=3,
)
```

The default backend is a compact, independently implemented STREME-inspired discovery algorithm. It supports variable-length seqlets and does not call STREME or FIMO. Use the same scorer as in `generate_designs`; the optimization objective stored with the designs defines the attribution target for single-task, multi-task, and combined-model predictions.

## Attribution and fragments

```text
Unique designs → ISM contributions → positive tangermeme seqlets
               → matched shuffled controls → motif discovery → catalog.meme
```

ISM scores the three alternative bases at each position. The projected observed-base contribution is the original score minus the four-base mean prediction. Positive fragments are extracted with tangermeme's recursive seqlet caller. Discovery uses fragment sequences without attribution weighting.

## Built-in discovery

1. Generate one length- and dinucleotide-matched Euler-trail shuffle per seqlet.
2. Split independent parent sequences into training and held-out groups. All fragments from one parent stay together. Family labels can be supplied to the lower-level interface below.
3. Pool each k-mer with its reverse complement and count independent training groups carrying it. Select enriched seeds across the requested widths.
4. Refine each seed by alternating PWM scanning, training-only threshold optimization, and PWM re-estimation from one best aligned site per training group. Scores are log2 odds against training-control base frequencies, symmetrized across strands and rounded to six decimal places for reproducible score ties.
5. Select the strongest training candidate, mask its matching training sites in foreground and controls, and continue. Neither held-out results nor retention decisions determine subsequent candidates or stopping.
6. For each fixed PWM, calibrate its reporting threshold on the original training fragments. Test that fixed threshold on held-out groups using a one-sided exact paired binomial test of foreground-only versus control-only group hits.
7. Correct held-out p-values by the configured `max_motifs` upper bound. Retain motifs with adjusted p ≤ `alpha`, at least `min_seqlets` fragments, and at least `min_parents` unique parents.

Invalid padding cannot enter a scanned window. Shorter fragments therefore contribute only where a motif fits. Informative flank differences remain in separate PWM candidates; the downstream selector accounts for their continuous PWM similarity.

This is an independent discovery method. Its paired group test and order-0 background differ from official STREME. `heldout_p_adjusted` refers to the matched shuffled-fragment null, while functional relevance is evaluated through the prediction model and experimental measurements.

## Parameters and outputs

```python
motifs = call_motifs(
    designs, model=scorer, motif_backend="builtin",
    output_dir="outputs/motifs_custom",
    seqlet_threshold=0.01, min_width=6, max_width=15,
    max_seqlet_len=25, flanks=2, max_motifs=30,
    min_seqlets=10, min_parents=3, seed=42,
    discovery_options={
        "per_width": 3, "iterations": 10,
        "holdout_fraction": 0.2, "alpha": 0.05,
    },
)
print(motifs.support)
```

At least ten independent groups are required, with at least three groups in each split. The default hold-out fraction is 20%. `max_motifs` limits the number of training candidates and defines the validation correction; the retained count follows the data.

Outputs include `contributions.npz`, `seqlets.csv/fa`, `background.fa`, `discovery_split.csv`, `candidate_motifs.meme`, `support.csv`, and the retained `catalog.meme`. The support table includes consensus, width, threshold, training search statistics, held-out p-values, fragment support, parent support, and the retention decision. `config.json` records the backend and requested settings.

## Official STREME backend

```python
motifs = call_motifs(
    designs, model=scorer, motif_backend="streme",
    output_dir="outputs/external_streme",
)
```

This mode checks STREME and FIMO before attribution. Missing tools produce the installation command:

```bash
conda install -c conda-forge -c bioconda meme
```

Executable paths can be supplied as `streme="/path/to/streme"` and `fimo="/path/to/fimo"`. The external backend discovers motifs with STREME and counts supporting seqlets with FIMO. Its native discovery statistics and support criteria remain separate from the built-in paired-group validation.

## Existing contributions or seqlets

Projected ISM or DeepLIFTShap contributions can be supplied directly:

```python
motifs = call_motifs(designs, contributions=contributions, output_dir="outputs/from_scores")
```

An array follows input design row order, with one column per base. A DataFrame indexed by `sequence_id` provides explicit alignment. Exact DNA duplicates are combined internally.

To examine discovery on already-called seqlets, the lower-level method accepts a table with `dna` and `sequence_id` columns:

```python
from sequence_optimization.simple_streme import discover_seqlet_motifs
from sequence_optimization.motifs import write_meme

result = discover_seqlet_motifs(
    seqlets, groups=seqlets["sequence_family"],
    min_seqlets=10, min_parents=3, seed=42,
)
retained = result.support.loc[result.support.retained, "motif_id"]
write_meme("my_motifs.meme", {key: result.pwms[key] for key in retained})
```

`groups` supplies one independent-family label per seqlet; omit it to group by parent DNA. A given parent must belong to one group. `validation_mask` can provide a fixed group-respecting held-out split. Candidate PWMs and retention flags are returned separately for inspection.

## Reuse a MEME catalog for selection

```python
from sequence_optimization import select_candidates
selected = select_candidates(
    designs, motifs="outputs/motifs/catalog.meme", top_n=10,
    output_dir="outputs/panel_from_meme",
)
```

Replace that path with a catalog produced by another discovery method. Supply DNA letter-probability matrices with `ALPHABET= ACGT` and unique motif IDs. Selection uses FIMO to retain motif instances for composition and positional diversity; this stage still requires FIMO. Reusing a catalog as an in-memory object is also supported with `call_motifs(designs, catalog="catalog.meme")`.

## Implementation reference

The seed/refinement/erasure organization was informed by the STREME paper ([Bailey, 2021](https://doi.org/10.1093/bioinformatics/btab203)) and the independently developed [pystreme](https://github.com/vmindel/pystreme) implementation, reviewed at commit `dd5d3c52a9d7264b50bf43e2457e90a0a46e132a`. The implementation here uses NumPy/Numba over variable-length fragments, parent/family statistical units, and an independently specified paired validation procedure. It does not depend on pystreme.
