# Optimize a promoter with your own model

The trainable insert is 80 nt. Use [the PyTorch adapter](../pytorch-models.md) to append the fixed flanks expected by the model.

```python
import torch
import torch.nn.functional as F
from MotifSift import SequencePredictor, call_motifs, generate_designs, select_candidates

LEFT = "TGCATTTTTTTCACATC"
RIGHT = "GGTTACGGCTGTT"


def encode(sequence, probabilities):
    indices = torch.tensor(["ACGT".index(base) for base in sequence], device=probabilities.device)
    return F.one_hot(indices, num_classes=4).to(probabilities).expand(len(probabilities), -1, -1)


class PromoterPredictor(SequencePredictor):
    def prepare_input(self, probabilities):
        left = encode(LEFT, probabilities)
        right = encode(RIGHT, probabilities)
        return torch.cat([left, probabilities, right], dim=1).transpose(1, 2)
```

```python
model = PromoterPredictor(my_model, tasks=["promoter_activity"])
designs = generate_designs(model, length=80, objective="maximize")
motifs = call_motifs(designs, model)
panel = select_candidates(designs, motifs=motifs, top_n=10)
panel[["sequence", "score", "present_motifs"]]
```
