# MotifSift

Design sequences with a trained predictor, discover motifs in the candidates, and select a panel.

![MotifSift workflow](Fig.png)

## Install

```bash
git clone https://github.com/UTR-CODE/bioseq-optimization.git
cd bioseq-optimization
python -m pip install '.[motifs]'
```

See [installation](docs/installation.md) for MEME Suite and the pretrained models.

## Design, discover motifs, and select

Use a predictor from `bioseq_models` directly. For your own PyTorch model, use [SequencePredictor](docs/pytorch-models.md).

```python
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, select_candidates

model = load_model("optimus5prime")
designs = generate_designs(model, length=50, objective="maximize", gain_fractions=(0.5, 0.9))
designs = designs.loc[designs["gain_fraction"].eq(0.9)].copy()
motifs = call_motifs(designs, model, reverse_complement=False)
panel = select_candidates(designs, motifs=motifs, top_n=10)
```

## Examples

| Model | Notebook |
|---|---|
| Optimus 5-Prime | [5′ UTR design](examples/single-task.ipynb) |
| DeepSTARR | [Multi-output enhancer design](examples/multi-task.ipynb) |
| MPRALegNet | [Cell-selective enhancer design](examples/multiple-models.ipynb) |
| Saluki | [Synonymous eGFP CDS design](examples/saluki-egfp.ipynb) |

Use your own PyTorch model in the [single-output model](examples/custom-single-output.ipynb), [multi-output model](examples/custom-multi-output.ipynb), or [multiple independent models](examples/custom-independent-models.ipynb) notebook.

[Objectives and API](docs/core-api.md) · [Motif discovery](docs/motif-discovery.md) · [Panel selection](docs/motif-diversity.md)
