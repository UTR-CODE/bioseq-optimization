import numpy as np
import pytest

from BioSeqBench.optimization.core import (
    BudgetExceeded, CachedScorer, CallableScorer, OptimizationConfig,
    ScoreBatch, SequenceSpace, available_optimizers, run_optimizer,
    TorchSequenceScorer,
)


class HammingScorer:
    def __init__(self):
        self.calls = 0

    def score(self, sequences):
        values = np.asarray(sequences, dtype=str)
        self.calls += len(values)
        return ScoreBatch(np.asarray([value.count("G") for value in values], dtype=float))


def test_sequence_space_round_trip_and_mutants():
    space = SequenceSpace(4, "ACGT")
    assert space.decode(space.encode(["ACGT"])).tolist() == ["ACGT"]
    mutants = space.single_mutants(["ACGT"])
    assert len(mutants) == 12
    assert len(set(mutants.tolist())) == 12
    assert all(sum(a != b for a, b in zip(value, "ACGT")) == 1 for value in mutants)


def test_cached_scorer_counts_unique_queries_and_budget():
    base = HammingScorer()
    scorer = CachedScorer(base, budget=2)
    first = scorer.score(["AAAA", "GGGG"])
    second = scorer.score(["GGGG", "AAAA"])
    assert first.scores.tolist() == [0.0, 4.0]
    assert second.scores.tolist() == first.scores[::-1].tolist()
    assert scorer.query_count == 2 and base.calls == 2
    with pytest.raises(BudgetExceeded):
        scorer.score(["CCCC"])


def test_registered_optimizers_have_common_io_and_budget():
    space = SequenceSpace(6, "ACGT")
    initial = np.asarray(["A" * 6, "C" * 6, "G" * 6, "T" * 6])
    config = OptimizationConfig(budget=40, num_candidates=4, mutation_rate=0.1)
    scorer = CallableScorer(lambda values: np.asarray([value.count("G") for value in values]))
    for name in ["random", "ism_greedy", "simulated_annealing", "genetic_algorithm", "adalead"]:
        result = run_optimizer(name, initial, scorer, space, config, seed=17)
        assert result.algorithm == name
        assert len(result.sequences) == len(result.scores) == 4
        assert len(set(result.sequences.tolist())) == 4
        assert result.query_count <= config.budget
        assert space.validate(result.sequences).shape == (4,)
    assert {"random", "ism_greedy", "simulated_annealing", "genetic_algorithm", "adalead"} <= set(available_optimizers())


def test_torch_sequence_scorer_adapter():
    import torch

    class Model(torch.nn.Module):
        def forward(self, inputs):
            return inputs.sum(dim=(1, 2))

    def encode(values):
        return torch.tensor([[[float(char == "G") for char in value] for _ in range(1)] for value in values])

    scorer = TorchSequenceScorer(Model(), encode, device="cpu", batch_size=2)
    result = scorer.score(["GG", "AA", "AG"])
    assert result.scores.tolist() == [2.0, 0.0, 1.0]
    assert result.uncertainties.tolist() == [0.0, 0.0, 0.0]
