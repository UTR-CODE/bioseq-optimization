# Algorithms and stopping

[Home](../README.md) · [Candidate generation](candidate-generation.md)

| Algorithm | API name | Input gradients |
|---|---|---|
| AdaLead | `adalead` | — |
| Base search | `basesearch` | — |
| Beam search | `beam_search` | — |
| Cross-entropy method | `cross_entropy_method` | — |
| Deep exploration network | `den` | Required |
| FastSeqProp | `fast_seqprop` | Required |
| Genetic algorithm | `genetic_algorithm` | — |
| Greedy ISM | `ism_greedy` | — |
| Ledidi | `ledidi` | Required |
| Simulated annealing | `simulated_annealing` | — |
| Activation maximization | `vanilla_activation_maximization` | Required |

`generate_designs` uses these 11 methods by default. Set `algorithms=[...]` to choose a subset. The registry also includes `random` for a separately budgeted random-search reference.

## Plateau stopping

```python
from sequence_optimization import SaturationConfig
saturation = SaturationConfig(
    fractions=(0.9,),
    selection="fraction",
    maximum_queries=None,
    material_improvement=0.01,
    relative_material_improvement=0.001,
)
```

The controller tracks archive quality and stops after sustained small gains. Default minimum and patience windows scale with population size and retained candidate count. Proposal-batch and round-based windows accommodate methods with large neighborhoods.

Each run records the query trajectory and stopping reason. The retained 90% observed-gain stage supplies 100 candidates for the combined pool. Knee-based selection is also available through `SaturationConfig(selection="knee")`.

ISM tracks active lineages separately. A lineage reaching a plateau leaves the active search population, while the remaining lineages continue. Optional native sequence kernels accelerate neighborhood construction.

Implementation: [algorithms.py](../sequence_optimization/algorithms.py), [advanced.py](../sequence_optimization/advanced.py), [saturation.py](../sequence_optimization/saturation.py).
