# K562 reference inputs

These are predictions and motif scans from the local study, not wet-lab measurements.

- `k562_candidates_1100.csv`: 11 non-random algorithms ×100 source records,1084 unique200-nt DNA sequences.
- `k562_catalog.meme`:11 retained DNA PWM models from the study-specific seqlet-derived catalog.
- `k562_fimo_sites.csv`:12164 physical sites passing the original scan p/q thresholds and same-ID footprint NMS. Original full-catalog q-values are retained.
- `k562_expected_top10.csv`: expected ordered panel for the documented default selection parameters; use DNA and stable sequence IDs to compare.
- `manifest.json`: input checksums and provenance.

No model checkpoints, external database collections or experimental activity claims are included. This catalog is specific to the K562 example. See [the complete tutorial](../../docs/motif-diversity.md).
