# Reference motif-diversity selection

Select a panel from the bundled K562 designs and motif hits.

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
