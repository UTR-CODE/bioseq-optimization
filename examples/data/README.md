# K562 reference data

Inputs for the [motif-diversity workflow](../../docs/motif-diversity.md).

| File | Contents |
|---|---|
| `k562_candidates_1100.csv` | 11 algorithms × 100 candidates; 1,084 unique 200-nt sequences |
| `k562_catalog.meme` | 11 seqlet-derived motif models |
| `k562_fimo_sites.csv` | 12,164 sites after p/q filtering and footprint consolidation |
| `k562_expected_top10.csv` | Ordered reference panel |
| `manifest.json` | Data provenance and SHA256 checksums |

The source study used Malinois activity predictions for K562, HepG2, and SKNSH. Specificity is K562 activity minus the maximum off-target activity. Sequence IDs are SHA256-based identifiers. FIMO q-values retain the original catalog-wide scan values.

The [motif-discovery reference files](motif_discovery) provide 4,235 variable-length seqlets, matched controls, CD-HIT-family split labels, and outputs for the built-in/STREME/pystreme comparison. See the [method comparison](../../docs/motif-discovery-comparison.md).
