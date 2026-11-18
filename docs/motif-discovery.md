# Motif discovery from attribution seqlets

[Home](../README.md) · [Core API](core-api.md)

The second core function converts designed sequences and model contributions into a motif catalog:

```python
from sequence_optimization import call_motifs

motifs = call_motifs(
    designs, model=scorer, output_dir="outputs/motifs",
    min_seqlets=10, min_parents=3,
)
```

The default backend uses official XSTREME with MEME disabled (`--meme-nmotifs 0`) to discover motifs and FIMO to identify supporting seqlets. The optional `motif_backend="builtin"` selects the independently implemented method described below. Use the same scorer as in `generate_designs`; the optimization objective stored with the designs defines the attribution target for single-task, multi-task, and combined-model predictions.

## Attribution and fragments

```text
Unique designs → ISM contributions → positive tangermeme seqlets
               → matched shuffled controls → motif discovery → catalog.meme
```

ISM scores the three alternative bases at each position. The projected observed-base contribution is the original score minus the four-base mean prediction. Positive fragments are extracted with tangermeme's recursive seqlet caller. Discovery uses fragment sequences without attribution weighting.

## Optional DeepLiftShap attribution

ISM is the default and requires discrete predictions only. To use DeepLiftShap, keep the discovery and filtering stages unchanged:

```python
motifs = call_motifs(
    designs, model=scorer, attribution_method="deepliftshap",
    n_references=20, attribution_batch_size=64, seed=42,
    output_dir="outputs/motifs_deepliftshap",
)
```

The adapter uses `score_soft` with canonical `[batch, length, 4]` A/C/G/T probabilities and the same scalar objective as optimization. Each sequence receives 20 dinucleotide-shuffled references; scorer-supplied context remains fixed. `attribution_batch_size` counts sequence–reference pairs. The projected observed-base contributions feed the same positive-seqlet caller. `attribution.json` records reference settings, forward agreement and completeness errors.

DeepLiftShap uses tangermeme's supported PyTorch nonlinear modules. Each nonlinear operation needs its own module instance; functional activations and reused nonlinear modules may require an attribution adapter. For a custom scorer or objective, supply `attribution_model=scalar_model`: a PyTorch module accepting `[batch, 4, length]` A/C/G/T inputs and returning `[batch, 1]` values for the **complete scalar design objective**, including fixed context and off-target penalties. Its predictions are checked against `score`; per-reference completeness errors must be below 0.001. The user's original model is kept intact through a private copy. ISM works with any discrete scorer and does not need this adapter.

## Official XSTREME backend (default)

For RNA tasks, set `reverse_complement=False` in `call_motifs`. All backends then discover motifs in the supplied orientation: XSTREME uses `--dna2rna`, standalone STREME uses `--rna` on U-encoded input copies, and the built-in method omits reverse-complement seed pooling, reverse scoring and background symmetrization. FIMO receives `--norc` for both seqlet-support and candidate scans. The default `True` retains the DNA workflow. The scorer itself must be strand-specific; this flag does not modify model predictions or arbitrary user-supplied attribution scores.

Native RNA discovery files remain unchanged. A separate `*_dna_encoded.meme` file maps only alphabet/background notation from U to T for the A/C/G/T FIMO interface; motif IDs, matrices, `nsites` and discovery statistics are preserved. The returned `MotifCatalog` carries the orientation into candidate selection, where PWM comparisons also omit reverse complements. When loading an exported MEME path or PWM dictionary, pass `reverse_complement=False` explicitly. Scan and selection receipts record the mode. If you supply saved hits, use hits from a forward-only scan, including its own q-values; filtering reverse-strand rows from a two-strand scan is insufficient.

```python
motifs = call_motifs(
    designs, model=scorer, motif_backend="xstreme",
    output_dir="outputs/external_xstreme",
)
```

This mode checks XSTREME and FIMO before attribution. Missing tools produce the installation command:

```bash
conda install -c conda-forge -c bioconda meme
```

Executable paths can be supplied as `xstreme="/path/to/xstreme"` and `fimo="/path/to/fimo"`. The external backend discovers motifs with XSTREME and counts supporting seqlets with FIMO. Its native discovery statistics and support criteria remain separate from the built-in paired-group validation.

The default output directory contains `contributions.npz`, `seqlets.csv/fa`, `background.fa`, XSTREME output, seqlet FIMO matches, `support.csv`, and the retained `catalog.meme`.

## Candidate-wide site filtering

FIMO scans the full native discovery catalog against the complete unique-DNA candidate pool. Seqlet-supported motifs then retain hits with `p <= 0.001` and `q <= 0.1` (`p_threshold` and `q_threshold`). Q-values are kept from that scan, without recalculation after motif removal. Each motif must have **more than 30** distinct physical matches (`min_sites=30`); exact-DNA duplicates do not increase this count. Same-motif overlapping hits are consolidated before counting.

Motifs are then compared by their matched positions. On the same sequence, two footprints represent the same position when their overlap covers at least 80% of the shorter footprint; strand is ignored. If more than 80% of either motif's positions are covered by the other, only one is retained (`subset_threshold=0.8`). Representatives are considered in descending site count, then descending total PWM information content, then motif ID. Removed motifs cannot merge other groups through a chain of overlaps.

`motif_site_support.csv` records site counts, retention and the retained representative for removed motifs. `motif_redundancy.csv` records the directional coverage fractions for the representative comparisons. `catalog.meme` contains the final motifs, and `motif_sites.csv` records their retained positions. Native PWM matrices, `nsites`, background and discovery statistics are preserved in exported MEME catalogs. Preserving `nsites` matters because FIMO uses it when applying its motif pseudocount. Candidate hits are cached in the returned catalog for selection; FIMO q-values remain those of the original candidate scan.

```python
motifs = call_motifs(
    designs, model=scorer, motif_backend="xstreme",
    min_sites=30, subset_threshold=0.8,
    output_dir="outputs/motifs",
)
```

XSTREME runs STREME discovery, enrichment assessment and motif grouping with the MEME engine disabled. `max_motifs` limits the STREME candidates, rather than fixing the final catalog size. Discovery statistics remain available in the native output; FIMO site support is a separate criterion. Its native nonredundant motifs are followed by the candidate-position filter above. The previous standalone backend remains available through `motif_backend="streme"`.

## Optional built-in discovery

1. Generate one length- and dinucleotide-matched Euler-trail shuffle per seqlet.
2. Split independent parent sequences into training and held-out groups. All fragments from one parent stay together. Family labels can be supplied to the lower-level interface below.
3. Pool each k-mer with its reverse complement by default (`reverse_complement=True`) and count independent training groups carrying it. In forward-only mode, keep the supplied k-mers without pooling. Select enriched seeds across the requested widths.
4. Refine each seed by alternating PWM scanning, training-only threshold optimization, and PWM re-estimation from one best aligned site per training group. Scores are log2 odds against training-control base frequencies, symmetrized across strands and rounded to six decimal places for reproducible score ties.
5. Select the strongest training candidate, mask its matching training sites in foreground and controls, and continue. Neither held-out results nor retention decisions determine subsequent candidates or stopping.
6. For each fixed PWM, calibrate its reporting threshold on the original training fragments. Test that fixed threshold on held-out groups using a one-sided exact paired binomial test of foreground-only versus control-only group hits.
7. Correct held-out p-values by the configured `max_motifs` upper bound. Retain motifs with adjusted p ≤ `alpha`, at least `min_seqlets` fragments, and at least `min_parents` unique parents.

Invalid padding cannot enter a scanned window. Shorter fragments therefore contribute only where a motif fits. Informative flank differences remain in separate PWM candidates; the downstream selector accounts for their continuous PWM similarity.

This is an independent discovery method. Its paired group test and order-0 background differ from official STREME. `heldout_p_adjusted` refers to the matched shuffled-fragment null, while functional relevance is evaluated through the prediction model and experimental measurements.

### Built-in parameters and outputs

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
