# Reference motif-diversity selection

Reuse saved designs and motifs from the K562 example to select a panel with the current filtering rules. The archived Top10 used the earlier motif catalog and filtering rules.

```python
from pathlib import Path
import pandas as pd
from MotifSift import call_motifs, select_candidates

DATA = Path("data")
designs = pd.read_csv(DATA / "k562_candidates_1100.csv")
motifs = call_motifs(
    designs, catalog=DATA / "k562_catalog.meme",
    hits=pd.read_csv(DATA / "k562_fimo_sites.csv"),
)
selected = select_candidates(
    designs, motifs=motifs, top_n=10, min_score=20.,
    output_dir="outputs/reference_top10",
)
print(selected[["selection_rank", "sequence_id", "score", "present_motifs"]])
```
