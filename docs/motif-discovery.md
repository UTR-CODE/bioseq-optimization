# Motif discovery

```python
from MotifSift import call_motifs

motifs = call_motifs(designs, model)
```

MotifSift uses in silico mutagenesis to score nucleotide contributions to the design objective, extracts positive seqlets, discovers motifs with XSTREME, and checks their support with FIMO. For RNA, pass `reverse_complement=False` to analyze the supplied orientation.

The catalog includes the retained motif models, motif sites, and the supporting evidence in `motifs.support`. Use `output_dir` to save the catalog and analysis files. Pass `contributions=` to reuse previously computed nucleotide contributions or `catalog=` to work with an existing motif catalog.

## Candidate-wide site filtering

By default, FIMO hits satisfy `p ≤ 0.001` and `q ≤ 0.1`. Overlapping hits for the same motif are consolidated. Each retained motif requires more than 30 sites on unique candidate sequences. Motifs with more than 80% overlap in either site set are merged by retaining one representative.
