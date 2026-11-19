# Optimize a 5′ UTR with your own model

For pretrained Optimus 5-Prime, see the [executed workflow](../examples/single-task.md). For your own PyTorch model, use the same workflow with a `SequencePredictor`.

```python
from MotifSift import SequencePredictor, call_motifs, generate_designs, select_candidates

model = SequencePredictor(my_model, tasks=["mrl"])
designs = generate_designs(model, length=50, objective="maximize")
motifs = call_motifs(designs, model, reverse_complement=False)
panel = select_candidates(designs, motifs=motifs, top_n=10)
panel[["sequence", "score", "present_motifs"]]
```
