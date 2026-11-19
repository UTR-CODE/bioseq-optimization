# Reproduce the original Malinois K562 motif catalog

[Notebook](../../examples/enhancer.ipynb) · [Installation](../installation.md)

The paper workflow starts from the archived 1,100 optimizer records and their Malinois ISM contributions for 1,084 unique enhancers. These are the inputs in the separate K562 13-motif reproduction bundle. The example data used by the [motif diversity tutorial](../motif-diversity.md) come from a different run.

    export MOTIFSIFT_K562_ARCHIVE=/path/to/K562_13motifs_reproduction_20261119

Run the notebook with the historical seqlet caller and MEME Suite 5.5.9 on PATH. The notebook calls the current MotifSift library and runs single-pass XSTREME from 4,229 positive seqlets, followed by FIMO site support and spatial redundancy filtering. It checks all 9,464 physical sites against the archive and expects 13 final motifs. The panel uses the same 20th-percentile specificity cutoff as the study.

For an exact historical replay using the native discovery catalog and archived FIMO scans, run `python scripts/reproduce.py replay --output runs/replay` from the reproduction bundle. The bundle also includes the historical seqlet function and commands for reproducing discovery from frozen ISM.

The notebook's optional fresh-design cell uses FP32, base seed 42 and fixed batches with the current optimizers. The original candidates were generated using a shared GPU scoring service with dynamic batches. The exact paper replay uses the archived candidates and their original scores.
