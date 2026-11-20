# MotifSift

Design sequences with a trained predictor, discover motifs in the candidates, and select a panel.

![MotifSift workflow](Fig.png)

## Quickstart

```bash
git clone https://github.com/xinhe-bio/MotifSift.git
cd MotifSift
python -m pip install '.[motifs]'
conda install -c conda-forge -c bioconda meme
```

See [installation](docs/installation.md) for `bioseq_models`, which provides the pretrained model used below.

### Design, discover motifs, and select

Use a predictor from `bioseq_models` directly. For your own PyTorch model, use [SequencePredictor](docs/pytorch-models.md).

```python
from bioseq_models import load_model
from MotifSift import call_motifs, generate_designs, select_candidates

model = load_model("optimus5prime")
designs = generate_designs(model, length=50, objective="maximize")
motifs = call_motifs(designs, model, reverse_complement=False)
panel = select_candidates(designs, motifs=motifs, top_n=10)
```

## Examples

| Model | Notebook |
|---|---|
| DREAM LegNet | [Yeast promoter design](examples/dream-legnet-promoter.ipynb) |
| Optimus 5-Prime | [5′ UTR design](examples/single-task.ipynb) |
| Malinois | [K562 enhancer specificity](examples/enhancer.ipynb) |
| DeepSTARR | [Multi-output enhancer design](examples/multi-task.ipynb) |
| MPRALegNet | [Cell-selective enhancer design](examples/multiple-models.ipynb) |
| Saluki | [Synonymous eGFP CDS design](examples/saluki-egfp.ipynb) · [Synonymous Gluc CDS design](examples/saluki-gluc-cds.ipynb) · [Gluc 3′ UTR design](examples/saluki-gluc-3utr.ipynb) |

Use your own PyTorch model in the [single-output model](examples/custom-single-output.ipynb), [multi-output model](examples/custom-multi-output.ipynb), or [multiple independent models](examples/custom-independent-models.ipynb) notebook. For a model exposing `score()` and `score_soft()`, see the [custom scoring example](examples/custom-scoring-methods.ipynb).

See the [custom constraints example](examples/custom-constraints.ipynb) to filter patterns, GC content, and homopolymers or keep selected positions fixed.

[Objectives and API](docs/core-api.md) · [Gain fraction](docs/gain-fraction.md) · [Motif discovery](docs/motif-discovery.md) · [Panel selection](docs/motif-diversity.md)

[Reproduce the Extended Data design examples](docs/extended-data-figures.md)
