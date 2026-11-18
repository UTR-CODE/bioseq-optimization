# Optimize enhancer specificity with Malinois

Maximize predicted K562 activity minus the higher of the two off-target predictions.

```python
import torch
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, select_candidates

model = load_model("malinois")
```

```python
def specificity(y):
    return y["K562_log2FC"] - torch.maximum(y["HepG2_log2FC"], y["SKNSH_log2FC"])


designs = generate_designs(model, length=200, objective=specificity)
motifs = call_motifs(designs, model)
panel = select_candidates(designs, motifs=motifs, top_n=10)
panel[["sequence", "score", "present_motifs"]]
```
