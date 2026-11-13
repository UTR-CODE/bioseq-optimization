"""Exact single-nucleotide contributions for scalar sequence scorers."""

from dataclasses import dataclass

import numpy as np

from .api import SequenceSpace, normalize_nucleotide_sequences


@dataclass
class ISMContributions:
    sequences: np.ndarray
    reference_predictions: np.ndarray
    mutant_predictions: np.ndarray
    observed: np.ndarray
    logo: np.ndarray


def ism_contributions(sequences, score_fn, *, batch_size=256):
    """Score all 3*L substitutions and center four-base predictions per position.

    ``score_fn`` must return one finite scalar per sequence. For the actual base
    at position i, contribution = f(sequence) - mean_b f(sequence[i <- b]).
    The original-base prediction is reused, not scored as another mutation.
    ``logo`` masks centered scores to the actual base, in A/C/G/T channel order.
    These local substitution effects are neither additive output decompositions
    nor motif interaction measurements. They require no gradient approximation.
    """
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, (int, np.integer))
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    values = normalize_nucleotide_sequences(sequences)
    if not len(values):
        raise ValueError("at least one sequence is required")
    space = SequenceSpace(len(values[0]))
    codes = space.encode(values)

    def score(batch):
        result = np.asarray(score_fn(np.asarray(batch, dtype=str)), dtype=float)
        if result.shape == (len(batch), 1):
            result = result[:, 0]
        if result.shape != (len(batch),) or not np.isfinite(result).all():
            raise ValueError(
                "score_fn must return a finite [N] scalar prediction vector"
            )
        return result

    reference = np.concatenate(
        [score(values[i : i + batch_size]) for i in range(0, len(values), batch_size)]
    )
    mutants = np.empty((len(values), space.length, 4), dtype=float)
    for row, sequence in enumerate(values):
        mutants[row, np.arange(space.length), codes[row]] = reference[row]
        pending, indices = [], []
        for position, actual in enumerate(sequence):
            for base_index, base in enumerate("ACGT"):
                if base == actual:
                    continue
                pending.append(sequence[:position] + base + sequence[position + 1 :])
                indices.append((position, base_index))
                if len(pending) == batch_size:
                    predictions = score(pending)
                    for index, prediction in zip(indices, predictions):
                        mutants[(row, *index)] = prediction
                    pending, indices = [], []
        if pending:
            for index, prediction in zip(indices, score(pending)):
                mutants[(row, *index)] = prediction
    observed = reference[:, None] - mutants.mean(axis=2)
    logo = np.eye(4)[codes] * observed[:, :, None]
    return ISMContributions(values, reference, mutants, observed, logo)


def _conservative_maxpool1d(module, grad_input, grad_output):
    """DeepLIFT max-pool routing with symmetric zero-delta fallback.

    Adapted from tangermeme (MIT, Jacob Schreiber); see
    THIRD_PARTY_NOTICES.md. Scatter-add preserves overlapping pooling windows.
    """
    import torch
    from torch.nn import functional as F

    with torch.no_grad():
        x, reference = module.input.chunk(2)
        output, baseline = module.output.chunk(2)
        maximum = torch.maximum(output, baseline)
        shares = torch.cat((maximum - baseline, output - maximum))
        _, indices = F.max_pool1d(
            module.input, module.kernel_size, module.stride, module.padding,
            module.dilation, module.ceil_mode, return_indices=True)
        routed = torch.zeros_like(module.input)
        routed.scatter_add_(-1, indices, grad_output[0] * shares)
        a, b = routed.chunk(2)
        difference = x - reference
        unchanged = difference == 0
        half = torch.where(unchanged, grad_input[0].chunk(2)[0],
                           (a + b) / difference.masked_fill(unchanged, 1))
        return (torch.cat((half, half)),)


def deepliftshap_contributions(sequences, scorer, objective, *,
                              attribution_model=None, n_references=20,
                              batch_size=64, seed=42):
    """Optional tangermeme DeepLiftShap; return observed scores and diagnostics.

    The default adapter calls scorer.score_soft on canonical [N,L,4] inputs.
    An explicit attribution_model instead accepts [N,4,L] ACGT inputs and
    returns [N,1] containing the complete scalar design objective. Its forward
    predictions must agree with the discrete scorer. Nonlinear operations
    must be represented by separate supported torch modules for DeepLIFT.
    """
    import copy
    import torch
    from tangermeme.deep_lift_shap import deep_lift_shap
    from tangermeme.ersatz import dinucleotide_shuffle
    from .api import as_score_batch
    from .objectives import MultiTaskObjective
    from .onehot import onehot

    for name, value in [('n_references', n_references), ('batch_size', batch_size)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    values = normalize_nucleotide_sequences(sequences)
    if not len(values):
        raise ValueError('at least one sequence is required')
    task_names = getattr(scorer, 'task_names', None)
    raw = as_score_batch(scorer.score(values)).scores
    expected = objective.scalarize(raw, getattr(scorer, 'task_names', task_names))

    if attribution_model is None:
        if not callable(getattr(scorer, 'score_soft', None)):
            raise TypeError('DeepLiftShap requires score_soft or attribution_model; ISM only needs score')
        # Work on a private copy: attribution hooks and device/dtype changes
        # must not alter the user's optimization model.
        adapted = copy.deepcopy(scorer)

        class ScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scorer = adapted
                modules, visited = [], set()

                def collect(value):
                    if id(value) in visited:
                        return
                    visited.add(id(value))
                    if isinstance(value, torch.nn.Module):
                        modules.append(value)
                    else:
                        for name in ('model', 'scorer'):
                            child = getattr(value, name, None)
                            if child is not None:
                                collect(child)
                        for child in getattr(value, 'models', ()):
                            collect(child)

                collect(adapted)
                self.networks = torch.nn.ModuleList(modules)
                if not modules:
                    raise TypeError('Cannot locate torch modules; supply attribution_model explicitly')

            def forward(self, x):
                output = self.scorer.score_soft(x.transpose(1, 2))
                activities = output[0] if isinstance(output, tuple) else output
                if activities.ndim == 1:
                    activities = activities[:, None]
                names = getattr(self.scorer, 'task_names', None)
                if isinstance(objective, MultiTaskObjective):
                    scalar = objective.scalarize_soft(activities, names)
                elif callable(getattr(objective, 'coefficients', None)):
                    weights = torch.as_tensor(objective.coefficients(activities.shape[1], names),
                                              dtype=activities.dtype, device=activities.device)
                    scalar = activities @ weights
                else:
                    raise TypeError('Supply attribution_model for a custom nonlinear objective')
                return scalar[:, None]

        model = ScalarModel()
    else:
        if not isinstance(attribution_model, torch.nn.Module):
            raise TypeError('attribution_model must be a torch.nn.Module')
        model = copy.deepcopy(attribution_model)
    model.eval()
    tensor = next(model.parameters(), None)
    if tensor is None:
        tensor = next(model.buffers(), None)
    device = tensor.device if tensor is not None else torch.device('cpu')
    dtype = tensor.dtype if tensor is not None and tensor.is_floating_point() else torch.float32
    x = torch.as_tensor(onehot(values, channels_first=True), dtype=dtype)
    observed, residuals = [], []
    precision_retries = 0
    rng = np.random.RandomState(seed)
    max_forward_error = 0.
    for start in range(0, len(x), 8):
        block = x[start:start+8]
        refs = dinucleotide_shuffle(block, n=n_references, random_state=rng)
        with torch.no_grad():
            prediction = model(block.to(device)).detach().cpu()
            if prediction.shape != (len(block), 1):
                raise ValueError('attribution_model must return [N,1] scalar objective values')
            error = np.max(np.abs(prediction[:, 0].numpy() - expected[start:start+len(block)]))
            max_forward_error = max(max_forward_error, float(error))
            if not np.allclose(prediction[:, 0].numpy(), expected[start:start+len(block)], atol=1e-4, rtol=1e-5):
                raise ValueError('DeepLiftShap forward differs from the design objective; supply an equivalent attribution_model')
            reference_predictions = torch.cat([
                model(part.to(device)).detach().cpu()
                for part in refs.reshape(-1, 4, x.shape[-1]).split(batch_size)
            ]).reshape(len(block), n_references)
        multipliers = deep_lift_shap(
            model, block, references=refs, target=0, batch_size=batch_size,
            raw_outputs=True, device=device, warning_threshold=.001, verbose=False,
            additional_nonlinear_ops={torch.nn.MaxPool1d: _conservative_maxpool1d})
        pair_effect = (multipliers * (block[:, None] - refs)).sum((2, 3))
        residual = (pair_effect - (prediction - reference_predictions)).abs()
        if torch.isfinite(multipliers).all() and residual.max() >= .001 and dtype != torch.float64:
            # Recompute numerically unstable example/reference pairs in float64.
            bad = torch.nonzero(residual >= .001)
            double_model = copy.deepcopy(model).double()
            bx = torch.stack([block[int(i)] for i, j in bad]).double()
            br = torch.stack([refs[int(i), int(j)] for i, j in bad])[:, None].double()
            corrected = deep_lift_shap(
                double_model, bx, references=br, target=0, batch_size=batch_size,
                raw_outputs=True, device=device, warning_threshold=.001, verbose=False,
                additional_nonlinear_ops={torch.nn.MaxPool1d: _conservative_maxpool1d})
            for k, (i, j) in enumerate(bad):
                multipliers[int(i), int(j)] = corrected[k, 0].to(multipliers.dtype)
            precision_retries += len(bad)
            del double_model
            pair_effect = (multipliers * (block[:, None] - refs)).sum((2, 3))
            residual = (pair_effect - (prediction - reference_predictions)).abs()
        if not torch.isfinite(multipliers).all() or not torch.isfinite(residual).all() or residual.max() >= .001:
            raise ValueError(
                'DeepLiftShap completeness error >= 0.001. Use separate supported '
                'nonlinear modules in attribution_model, or use ISM.')
        hypothetical = (multipliers - (multipliers * refs).sum(2, keepdim=True)).mean(1)
        observed.append((hypothetical * block).sum(1).numpy())
        residuals.append(float(residual.max()))
    return np.concatenate(observed), dict(
        method='deepliftshap', n_references=n_references,
        reference='dinucleotide-shuffled design; scorer context fixed', seed=seed,
        pair_batch_size=batch_size, max_completeness_error=max(residuals),
        max_forward_error=max_forward_error, precision_retried_pairs=precision_retries)
