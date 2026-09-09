# Single-output model compatibility, 2026-09-09

All **32 registered single-output models pass all eight black-box algorithms**.
Of these, 28 also pass all three gradient algorithms with the existing
`DifferentiableTorchSequenceScorer` and the tensor transforms in the test script.
Cetnar requires CPU in the tested configuration. Four models still need a
model-specific differentiable adapter; their direct gradient calls reject the
missing `score_soft()` interface as expected.

| Models | Count | Black-box algorithms | Gradient algorithms |
|---|---:|---|---|
| Optimus 5-Prime | 1 | 8/8, direct | 3/3, wrapper, CUDA |
| DeepSTARR-Mouse, all six tissues × activity/accessibility | 12 | 8/8 each, direct | 3/3 each, wrapper, CUDA |
| DeepSTARR-Embryo, all registered tissue/output models | 11 | 8/8 each, direct | 3/3 each, wrapper, CUDA |
| Agarwal HepG2, K562, WTC11 | 3 | 8/8 each, direct | 3/3 each, wrapper, CUDA |
| Cetnar stability | 1 | 8/8, direct | 3/3 on CPU; CUDA/cuDNN backward fails in eval mode |
| DREAM LegNet | 1 | 8/8, direct | Requires differentiable context/channel adapter |
| PARM K562 | 1 | 8/8, direct | Requires differentiable adapter insertion and padding |
| UTR-CODE | 1 | 8/8, direct | Requires differentiable transcript/context adapter |
| CDS Stability | 1 | 8/8, direct | Requires handling codon tokenization and auxiliary inputs |

Black-box algorithms: `random`, `ism_greedy`, `simulated_annealing`,
`genetic_algorithm`, `adalead`, `beam_search`, `cross_entropy_method`, `cbas`.
Gradient algorithms: `vanilla_activation_maximization`, `fast_seqprop`, `den`.

## Evidence and reproduction

The CUDA matrix contains 352 entries: **337 passed**, 15 unavailable in the
tested configuration, zero other failures. A targeted Cetnar CPU run passes
all 11 algorithms, resolving three of the CUDA limitations.

- [Full CUDA matrix](single_task_model_results_2026-09-09.json)
- [Cetnar CPU results](single_task_cetnar_cpu_2026-09-09.json)
- [Reproducible runner](../scripts/check_single_task_models.py)

Run from the repository root with `bioseq-models` installed:

```bash
python scripts/check_single_task_models.py --device cuda --output matrix.json
python scripts/check_single_task_models.py --device cpu --models cetnar_stability --output cetnar_cpu.json
```

For a source checkout, put both the optimization repository and model repository
on `PYTHONPATH`. The recorded run used the UTR Python 3.10 environment and local
`bioseq-models` commit `f462dc7aa34ec4501afbbf99c9f46c250d589512`, with its real
pretrained checkpoints and original ensembles. DeepSTARR two-output and Malinois
three-output models were excluded by `len(model_info(name).task_names) == 1`.

## What was checked

Each run starts with four reproducible random sequences (`seed=7`). ISM receives
`4 + 3 * biological_length` queries to perform at least one complete single-base
mutant neighborhood. Other algorithms receive 12 queries. Generator pretraining
and update epochs are reduced to one for this smoke test.

Every successful run checks four unique ACGT sequences of the correct length,
finite and matching objective/activity values, correct task names, and query
count greater than the four initial seeds and within the configured budget.
Activities are independently recomputed with the original loaded model. For
gradient wrappers, hard one-hot predictions are first compared with the original
string scorer, and finite input gradients are checked before optimization.
Agarwal requires A/G/C/T channel order; Cetnar uses a last-channel tensor.

Fixed-length models use registry lengths (50, 110, 230, or 1001 nt). Variable
length cases use Cetnar 80 nt, PARM 50 nt, UTR-CODE 50-nt 5′UTR, and CDS Stability
60 nt. UTR-CODE uses the bundled HEK293T context, a fixed synthetic CDS and 3′UTR;
CDS Stability uses the default human species. These are representative test
inputs, not validation of every length, species, transcript region, or context.

This verifies execution and output consistency, not convergence, biological
validity, or ranking of algorithm effectiveness. In particular, the current
nucleotide search space does not preserve CDS amino-acid sequence or enforce an
open reading frame. Query counts follow the existing optimizer accounting;
they are not measurements of all forward/backward computation.

`unsupported` in the JSON means unavailable through the tested interface or
backend. It does not mean that a future differentiable adapter is impossible.
The model-specific transforms are test fixtures, not new public model adapters.
