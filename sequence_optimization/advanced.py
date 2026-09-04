"""Generative, beam, and differentiable sequence optimizers."""

from __future__ import annotations

import copy
import math

import numpy as np

from .algorithms import _finalize, _score, register_algorithm
from .api import CachedScorer, SequenceSpace
from .objectives import ObjectiveScorer


def _top_archive(
    archive: dict[str, tuple[float, float]],
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    ranked = sorted(archive.items(), key=lambda item: (-item[1][0], item[0]))[:count]
    return (
        np.asarray([item[0] for item in ranked], dtype=str),
        np.asarray([item[1][0] for item in ranked], dtype=np.float64),
    )


@register_algorithm("beam_search")
def beam_search(initial, scorer, space, config, rng):
    """Expand single mutants of the best current sequences as a shared beam."""
    archive: dict[str, tuple[float, float]] = {}
    initial_scores = _score(scorer, archive, initial).scores
    beam_width = max(1, int(round(config.num_candidates * config.elite_fraction)))
    order = np.argsort(initial_scores)[::-1][:beam_width]
    beam = initial[order]
    trajectory = []
    rounds = 0
    while scorer.remaining is None or scorer.remaining > 0:
        proposals = np.unique(space.single_mutants(beam))
        if scorer.remaining is not None and len(proposals) > scorer.remaining:
            chosen = rng.choice(len(proposals), size=scorer.remaining, replace=False)
            proposals = proposals[chosen]
        if not len(proposals):
            break
        before = scorer.query_count
        _score(scorer, archive, proposals)
        rounds += 1
        beam, beam_scores = _top_archive(archive, beam_width)
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(beam_scores.max()),
                "mean_score": float(beam_scores.mean()),
            }
        )
        if scorer.query_count == before:
            break
    return _finalize(
        "beam_search",
        archive,
        config,
        scorer,
        {"rounds": rounds, "beam_width": beam_width},
        trajectory,
    )


@register_algorithm("cross_entropy_method")
def cross_entropy_method(initial, scorer, space, config, rng):
    """Optimize a position-wise categorical generator with elite samples."""
    archive: dict[str, tuple[float, float]] = {}
    initial_batch = _score(scorer, archive, initial)
    encoded = space.encode(initial)
    probabilities = np.empty((space.length, space.alphabet_size), dtype=np.float64)
    for position in range(space.length):
        counts = np.bincount(encoded[:, position], minlength=space.alphabet_size) + 1.0
        probabilities[position] = counts / counts.sum()
    trajectory = [
        {
            "query_count": scorer.query_count,
            "best_score": float(initial_batch.scores.max()),
            "mean_score": float(initial_batch.scores.mean()),
        }
    ]
    rounds = 0
    smoothing = min(float(config.learning_rate), 1.0)
    while scorer.remaining is None or scorer.remaining > 0:
        count = (
            config.num_candidates
            if scorer.remaining is None
            else min(config.num_candidates, scorer.remaining)
        )
        draws = rng.random((count, space.length, 1))
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0
        samples = (draws > cumulative[None, :, :]).sum(axis=2).astype(np.int16)
        sequences = space.decode(samples)
        before = scorer.query_count
        batch = _score(scorer, archive, sequences)
        elite_count = max(1, int(round(count * config.elite_fraction)))
        elite = samples[np.argsort(batch.scores)[-elite_count:]]
        fitted = np.full_like(probabilities, 1e-3)
        for position in range(space.length):
            fitted[position] += np.bincount(
                elite[:, position], minlength=space.alphabet_size
            )
        fitted /= fitted.sum(axis=1, keepdims=True)
        probabilities = (1.0 - smoothing) * probabilities + smoothing * fitted
        rounds += 1
        top_sequences, top_scores = _top_archive(archive, config.num_candidates)
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(top_scores.max()),
                "mean_score": float(top_scores.mean()),
            }
        )
        if scorer.query_count == before:
            break
    return _finalize(
        "cross_entropy_method",
        archive,
        config,
        scorer,
        {"rounds": rounds, "smoothing": smoothing},
        trajectory,
    )


def _torch_device(scorer: CachedScorer):
    import torch

    objective = scorer.scorer
    raw = objective.scorer if isinstance(objective, ObjectiveScorer) else objective
    configured = getattr(raw, "device", None)
    return torch.device("cpu" if configured is None else configured)


def _fit_autoregressive(
    model,
    encoded,
    *,
    epochs: int,
    learning_rate: float,
    weights=None,
) -> None:
    import torch

    if epochs == 0 or len(encoded) == 0:
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        log_probability = model.log_prob(encoded)
        if weights is None:
            loss = -log_probability.mean()
        else:
            loss = -(weights * log_probability).sum() / weights.sum().clamp_min(1e-8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def _autoregressive_generator(length: int, alphabet_size: int, device):
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class AutoregressiveGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.start_token = alphabet_size
            self.embedding = nn.Embedding(alphabet_size + 1, 16)
            self.recurrent = nn.GRU(16, 64, batch_first=True)
            self.output = nn.Linear(64, alphabet_size)

        def logits(self, encoded):
            start = torch.full(
                (len(encoded), 1),
                self.start_token,
                dtype=torch.long,
                device=encoded.device,
            )
            inputs = torch.cat([start, encoded[:, :-1]], dim=1)
            hidden, _ = self.recurrent(self.embedding(inputs))
            return self.output(hidden)

        def log_prob(self, encoded):
            return -functional.cross_entropy(
                self.logits(encoded).transpose(1, 2), encoded, reduction="none"
            ).sum(dim=1)

        @torch.no_grad()
        def sample(self, count: int):
            token = torch.full(
                (count, 1), self.start_token, dtype=torch.long, device=device
            )
            hidden = None
            output = []
            for _ in range(length):
                state, hidden = self.recurrent(self.embedding(token), hidden)
                logits = self.output(state[:, -1])
                token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                output.append(token)
            return torch.cat(output, dim=1)

    return AutoregressiveGenerator().to(device)


@register_algorithm("cbas")
def cbas(initial, scorer, space, config, rng):
    """Condition an autoregressive sequence generator on high activity."""
    import torch

    archive: dict[str, tuple[float, float]] = {}
    initial_batch = _score(scorer, archive, initial)
    if scorer.remaining is not None and scorer.remaining == 0:
        return _finalize("cbas", archive, config, scorer, {"rounds": 0}, [])

    device = _torch_device(scorer)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    encoded = torch.as_tensor(space.encode(initial), dtype=torch.long, device=device)
    prior = _autoregressive_generator(space.length, space.alphabet_size, device)
    _fit_autoregressive(
        prior,
        encoded,
        epochs=config.generator_pretrain_epochs,
        learning_rate=config.generator_learning_rate,
    )
    search = copy.deepcopy(prior)
    threshold = float(np.quantile(initial_batch.scores, config.cbas_quantile))
    trajectory = []
    rounds = 0
    while scorer.remaining is None or scorer.remaining > 0:
        count = (
            config.cbas_batch_size
            if scorer.remaining is None
            else min(config.cbas_batch_size, scorer.remaining)
        )
        samples = search.sample(count)
        sequences = space.decode(samples.cpu().numpy())
        before = scorer.query_count
        batch = _score(scorer, archive, sequences)
        threshold = max(
            threshold, float(np.quantile(batch.scores, config.cbas_quantile))
        )
        standard_deviation = np.zeros_like(batch.scores)
        if batch.uncertainties is not None:
            standard_deviation = batch.uncertainties
        floor = max(float(np.median(standard_deviation)) * 0.1, 1e-3)
        standard_deviation = np.maximum(standard_deviation, floor)
        z_score = (batch.scores - threshold) / standard_deviation
        success_probability = 0.5 * (
            1.0 + np.asarray([math.erf(value / math.sqrt(2.0)) for value in z_score])
        )
        success_probability = np.clip(success_probability, 1e-12, 1.0)
        with torch.no_grad():
            log_prior = prior.log_prob(samples).cpu().numpy()
            log_search = search.log_prob(samples).cpu().numpy()
        log_weights = np.log(success_probability) + log_prior - log_search
        log_weights = np.clip(log_weights - log_weights.max(), -30.0, 0.0)
        weights = torch.as_tensor(
            np.exp(log_weights), dtype=torch.float32, device=device
        )
        _fit_autoregressive(
            search,
            samples,
            epochs=config.generator_update_epochs,
            learning_rate=config.generator_learning_rate,
            weights=weights,
        )
        rounds += 1
        top_sequences, top_scores = _top_archive(archive, config.num_candidates)
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(top_scores.max()),
                "mean_score": float(top_scores.mean()),
            }
        )
        if scorer.query_count == before:
            break
    return _finalize(
        "cbas",
        archive,
        config,
        scorer,
        {
            "rounds": rounds,
            "conditioning_quantile": config.cbas_quantile,
            "final_threshold": threshold,
            "generator": "autoregressive_gru",
        },
        trajectory,
    )


def _differentiable_objective(scorer: CachedScorer) -> ObjectiveScorer:
    objective = scorer.scorer
    if not isinstance(objective, ObjectiveScorer) or not hasattr(
        objective.scorer, "score_soft"
    ):
        raise TypeError(
            "this algorithm requires DifferentiableTorchSequenceScorer or an equivalent "
            "scorer implementing score_soft(probabilities)"
        )
    return objective


def _one_hot(initial, space: SequenceSpace, device):
    import torch
    from torch.nn import functional as functional

    encoded = torch.as_tensor(space.encode(initial), dtype=torch.long, device=device)
    return functional.one_hot(encoded, num_classes=space.alphabet_size).to(
        torch.float32
    )


def _straight_through(probabilities):
    import torch
    from torch.nn import functional as functional

    shape = probabilities.shape
    sampled = torch.multinomial(probabilities.reshape(-1, shape[-1]), 1).reshape(
        shape[:-1]
    )
    hard = functional.one_hot(sampled, num_classes=shape[-1]).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach(), sampled


def _gradient_result(
    name,
    archive,
    config,
    scorer,
    trajectory,
    metadata,
):
    metadata = {"gradient_steps": len(trajectory), **metadata}
    return _finalize(name, archive, config, scorer, metadata, trajectory)


@register_algorithm("vanilla_activation_maximization")
def vanilla_activation_maximization(initial, scorer, space, config, rng):
    """Optimize a soft position-weight matrix through a differentiable model."""
    import torch

    objective = _differentiable_objective(scorer)
    archive: dict[str, tuple[float, float]] = {}
    _score(scorer, archive, initial)
    device = _torch_device(scorer)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    initial_one_hot = _one_hot(initial, space, device)
    logits = torch.nn.Parameter(
        initial_one_hot * 1.5 + 0.05 * torch.randn_like(initial_one_hot)
    )
    optimizer = torch.optim.Adam([logits], lr=config.learning_rate)
    trajectory = []
    while scorer.remaining is None or scorer.remaining >= len(initial):
        optimizer.zero_grad(set_to_none=True)
        probabilities = torch.softmax(logits, dim=-1)
        utilities, _ = objective.score_soft(probabilities)
        (-utilities.mean()).backward()
        optimizer.step()
        hard = logits.detach().argmax(dim=-1).cpu().numpy()
        sequences = space.decode(hard)
        batch = _score(scorer, archive, sequences)
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(batch.scores.max()),
                "mean_score": float(batch.scores.mean()),
            }
        )
    return _gradient_result(
        "vanilla_activation_maximization",
        archive,
        config,
        scorer,
        trajectory,
        {"learning_rate": config.learning_rate},
    )


@register_algorithm("fast_seqprop")
def fast_seqprop(initial, scorer, space, config, rng):
    """Fast SeqProp with channel-wise instance normalization and sampling."""
    import torch

    objective = _differentiable_objective(scorer)
    archive: dict[str, tuple[float, float]] = {}
    _score(scorer, archive, initial)
    device = _torch_device(scorer)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    initial_one_hot = _one_hot(initial, space, device)
    logits = torch.nn.Parameter(
        initial_one_hot + 0.05 * torch.randn_like(initial_one_hot)
    )
    gamma = torch.nn.Parameter(
        torch.ones(len(initial), space.alphabet_size, device=device)
    )
    beta = torch.nn.Parameter(
        torch.zeros(len(initial), space.alphabet_size, device=device)
    )
    optimizer = torch.optim.Adam([logits, gamma, beta], lr=config.learning_rate)
    trajectory = []
    while scorer.remaining is None or scorer.remaining >= len(initial):
        optimizer.zero_grad(set_to_none=True)
        mean = logits.mean(dim=1, keepdim=True)
        variance = logits.var(dim=1, keepdim=True, unbiased=False)
        normalized = (logits - mean) / torch.sqrt(variance + 1e-5)
        probabilities = torch.softmax(
            normalized * gamma[:, None, :] + beta[:, None, :], dim=-1
        )
        straight_through, sampled = _straight_through(probabilities)
        utilities, _ = objective.score_soft(straight_through)
        (-utilities.mean()).backward()
        optimizer.step()
        sequences = space.decode(sampled.detach().cpu().numpy())
        batch = _score(scorer, archive, sequences)
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(batch.scores.max()),
                "mean_score": float(batch.scores.mean()),
            }
        )
    return _gradient_result(
        "fast_seqprop",
        archive,
        config,
        scorer,
        trajectory,
        {"learning_rate": config.learning_rate},
    )


@register_algorithm("den")
def den(initial, scorer, space, config, rng):
    """Train a noise-to-sequence generator through a differentiable model."""
    import torch
    from torch import nn
    from torch.nn import functional as functional

    objective = _differentiable_objective(scorer)
    archive: dict[str, tuple[float, float]] = {}
    _score(scorer, archive, initial)
    device = _torch_device(scorer)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    noise_dim = 16

    class DeepExplorationGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(noise_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, space.length * space.alphabet_size),
            )

        def forward(self, noise):
            return self.network(noise).reshape(
                len(noise), space.length, space.alphabet_size
            )

    generator = DeepExplorationGenerator().to(device)
    fixed_noise = torch.randn(len(initial), noise_dim, device=device)
    target = torch.as_tensor(space.encode(initial), dtype=torch.long, device=device)
    pretrainer = torch.optim.Adam(
        generator.parameters(), lr=config.generator_learning_rate
    )
    for _ in range(config.den_seed_pretrain_epochs):
        reconstruction_loss = functional.cross_entropy(
            generator(fixed_noise).reshape(-1, space.alphabet_size), target.reshape(-1)
        )
        pretrainer.zero_grad(set_to_none=True)
        reconstruction_loss.backward()
        pretrainer.step()
    with torch.no_grad():
        reconstruction = generator(fixed_noise).argmax(dim=-1)
        reconstruction_accuracy = float((reconstruction == target).float().mean().cpu())

    optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.generator_learning_rate
    )
    trajectory = []
    while scorer.remaining is None or scorer.remaining >= len(initial):
        logits = generator(fixed_noise)
        probabilities = torch.softmax(logits, dim=-1)
        straight_through, sampled = _straight_through(probabilities)
        utilities, _ = objective.score_soft(straight_through)
        half = len(initial) // 2
        if half:
            pair_similarity = (
                (straight_through[:half] * straight_through[half : 2 * half])
                .sum(dim=-1)
                .mean()
            )
        else:
            pair_similarity = torch.zeros((), device=device)
        loss = -utilities.mean() + config.den_diversity_weight * pair_similarity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        optimizer.step()
        sequences = space.decode(sampled.detach().cpu().numpy())
        batch = _score(scorer, archive, sequences)
        trajectory.append(
            {
                "query_count": scorer.query_count,
                "best_score": float(batch.scores.max()),
                "mean_score": float(batch.scores.mean()),
            }
        )
    return _gradient_result(
        "den",
        archive,
        config,
        scorer,
        trajectory,
        {
            "generator": "deep_exploration_network",
            "noise_dim": noise_dim,
            "diversity_weight": config.den_diversity_weight,
            "seed_reconstruction_accuracy": reconstruction_accuracy,
        },
    )
