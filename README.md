# MotifSift

Design sequences with a trained predictor, discover motifs in the candidates, and select a panel for experimental follow-up.

![MotifSift workflow](Fig.png)

## Install

```bash
git clone https://github.com/UTR-CODE/bioseq-optimization.git
cd bioseq-optimization
python -m pip install '.[motifs]'
```

See [installation](docs/installation.md) for MEME Suite and the pretrained models used in the notebooks.

## Design, discover motifs, and select

Start with a trained [sequence-to-activity model](docs/pytorch-models.md) or [custom scorer](docs/custom-scorers.md).

```python
from sequence_optimization import call_motifs, generate_designs, select_candidates

designs = generate_designs(
    model, length=200, objective=["maximize"],
    gain_fractions=(0.5, 0.9),
)

candidates = designs.loc[designs["gain_fraction"].eq(0.9)]
motifs = call_motifs(candidates, model=model)
selected = select_candidates(candidates, top_n=10, motifs=motifs)
```

## Examples

| Model | Notebook |
|---|---|
| Optimus 5-Prime | [Optimize a 5′ UTR for predicted ribosome load](examples/single-task.ipynb) |
| DeepSTARR | [Optimize developmental and housekeeping enhancer activity](examples/multi-task.ipynb) |
| MPRALegNet | [Optimize enhancer activity across K562, HepG2 and WTC11](examples/multiple-models.ipynb) |
| Saluki | [Optimize the eGFP CDS with synonymous substitutions](examples/saluki-egfp.ipynb) |

[API](docs/core-api.md) · [Motif discovery](docs/motif-discovery.md) · [Panel selection](docs/motif-diversity.md)
