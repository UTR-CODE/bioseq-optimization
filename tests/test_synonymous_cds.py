import numpy as np
import pytest

from sequence_optimization import (
    CallableScorer,
    SaturationConfig,
    SequencePredictor,
    available_algorithms,
    basesearch_optimize,
    genetic_algorithm_optimize,
)
from sequence_optimization.codons import GENETIC_CODE, STOP_CODONS, SynonymousCodonSpace


def _protein(sequence):
    return "".join(
        GENETIC_CODE[sequence[index : index + 3]]
        for index in range(0, len(sequence), 3)
    )


def test_single_cds_saturation_preserves_translation():
    initial = "ATG" + "GCT" * 4 + "TAA"
    space = SynonymousCodonSpace(initial)

    def score(values):
        space.validate(values)
        return np.asarray([sequence.count("G") for sequence in values])

    result = genetic_algorithm_optimize(
        CallableScorer(score),
        length=len(initial),
        initial=[initial],
        objective=["maximize"],
        num_candidates=4,
        is_CDS_optimized=True,
        diversity=0,
        saturation=SaturationConfig(
            minimum_queries=4, plateau_patience_queries=4
        ),
    )
    for selection in result.selections.values():
        space.validate(selection.sequences)


def test_cds_random_candidates_use_bounded_synonymous_substitutions():
    reference = "ATG" + "GCT" * 8 + "TAA"
    space = SynonymousCodonSpace(reference)
    candidates = space.random(200, np.random.default_rng(13))
    codon_count = len(reference) // 3
    changes = [
        sum(
            candidate[index : index + 3] != reference[index : index + 3]
            for index in range(0, len(reference), 3)
        )
        for candidate in candidates
    ]

    assert all(0 < count < codon_count - 1 for count in changes)
    assert all(candidate.startswith("ATG") for candidate in candidates)
    assert all(_protein(candidate) == _protein(reference) for candidate in candidates)


def test_multiple_cds_are_validated_and_optimized_independently():
    initial = ["ATGGCTTAA", "ATGGATTAG"]
    queried = []

    def score(values):
        queried.extend(map(str, values))
        return np.asarray([value.count("G") for value in values])

    results = genetic_algorithm_optimize(
        CallableScorer(score),
        length=9,
        initial=initial,
        objective=["maximize"],
        num_candidates=2,
        is_CDS_optimized=True,
        diversity=0,
        mutation_rate=0.5,
        seed=7,
    )
    assert isinstance(results, list)
    assert len(results) == len(initial)
    for reference, result in zip(initial, results):
        assert len(result.sequences) == 2
        assert all(_protein(value) == _protein(reference) for value in result.sequences)
        assert all(value[:3] == "ATG" for value in result.sequences)
        assert all(value[-3:] in STOP_CODONS for value in result.sequences)
    assert {_protein(value) for value in queried} == {
        _protein(value) for value in initial
    }


def test_cds_fixed_region_is_preserved():
    initial = "ATG" + "GCT" * 8 + "TAA"
    queried = []

    def score(values):
        queried.extend(map(str, values))
        return np.asarray([value.count("G") for value in values])

    result = genetic_algorithm_optimize(
        CallableScorer(score),
        length=len(initial),
        initial=[initial],
        objective=["maximize"],
        num_candidates=4,
        is_CDS_optimized=True,
        fixed_region=["5:6"],
        diversity=0,
        mutation_rate=0.3,
        seed=7,
    )
    assert len(queried) > 4
    assert all(value[5] == initial[5] for value in queried)
    assert all(_protein(value) == _protein(initial) for value in queried)
    assert all(_protein(value) == _protein(initial) for value in result.sequences)


@pytest.mark.parametrize("stop_codon", sorted(STOP_CODONS))
def test_all_standard_terminal_stop_codons_are_accepted(stop_codon):
    initial = "ATGGCT" + stop_codon
    space = SynonymousCodonSpace(initial)
    candidates = ["ATGGCTTAA", "ATGGCCTAG", "ATGGCATGA"]
    assert space.validate(candidates).tolist() == candidates


def test_cds_candidate_validation_checks_boundaries_and_translation():
    space = SynonymousCodonSpace("ATGGCTTAA")
    with pytest.raises(ValueError, match="begin with ATG"):
        space.validate(["GTGGCTTAA"])
    with pytest.raises(ValueError, match="end with TAA, TAG, or TGA"):
        space.validate(["ATGGCTAAA"])
    with pytest.raises(ValueError, match="preserve the initial protein"):
        space.validate(["ATGGATTAA"])


@pytest.mark.parametrize(
    "initial",
    ["ATGGCT", "GTGGCTTAA", "ATGGCTAAA", "ATGTAGGCTTAA", "ATGGCNTAA"],
)
def test_invalid_initial_cds(initial):
    with pytest.raises(ValueError):
        SynonymousCodonSpace(initial)


def test_cds_optimizer_validation():
    model = CallableScorer(lambda values: np.zeros(len(values)))
    common = dict(objective=["maximize"], num_candidates=2)
    with pytest.raises(ValueError, match="initial is required"):
        genetic_algorithm_optimize(
            model, length=9, initial=None, is_CDS_optimized=True, **common
        )
    with pytest.raises(ValueError, match="length must match"):
        genetic_algorithm_optimize(
            model,
            length=12,
            initial=["ATGGCTTAA"],
            is_CDS_optimized=True,
            **common,
        )
    result = basesearch_optimize(
        model,
        length=9,
        initial=["ATGGCTTAA"],
        is_CDS_optimized=True,
        diversity=0,
        **common,
    )
    assert all(_protein(value) == _protein("ATGGCTTAA") for value in result.sequences)
    with pytest.raises(ValueError, match="complete ACGT CDS in frame"):
        genetic_algorithm_optimize(
            model,
            length=8,
            initial=["ATGGCTAA"],
            is_CDS_optimized=True,
            **common,
        )


@pytest.mark.parametrize("algorithm", available_algorithms())
def test_every_optimizer_preserves_the_initial_cds_translation(algorithm):
    import torch

    class GCount(torch.nn.Module):
        def forward(self, values):
            return values[:, 2].sum(dim=1)

    reference = "ATG" + "GCT" * 4 + "TAA"
    function_name = {
        "ism_greedy": "ism_optimize",
        "cross_entropy_method": "cross_entropy_optimize",
    }.get(algorithm, algorithm + "_optimize")
    import sequence_optimization as so

    result = getattr(so, function_name)(
        SequencePredictor(GCount()),
        length=len(reference),
        initial=[reference],
        objective=["maximize"],
        num_candidates=2,
        is_CDS_optimized=True,
        fixed_region=[5],
        diversity=0,
        generator_pretrain_epochs=1,
        generator_update_epochs=1,
        den_seed_pretrain_epochs=1,
        seed=7,
    )
    assert all(_protein(value) == _protein(reference) for value in result.sequences)
    assert all(value.startswith("ATG") for value in result.sequences)
    assert all(value[-3:] in STOP_CODONS for value in result.sequences)
    assert all(value[5] == reference[5] for value in result.sequences)


def test_every_cds_is_validated_before_independent_searches_start():
    queried = []
    model = CallableScorer(
        lambda values: queried.extend(values) or np.zeros(len(values))
    )
    with pytest.raises(ValueError, match="end with TAA, TAG, or TGA"):
        genetic_algorithm_optimize(
            model,
            length=9,
            initial=["ATGGCTTAA", "ATGGCTAAA"],
            objective=["maximize"],
            num_candidates=2,
            is_CDS_optimized=True,
            diversity=0,
        )
    assert queried == []


def test_cds_capacity_must_cover_candidate_count():
    model = CallableScorer(lambda values: np.zeros(len(values)))
    with pytest.raises(ValueError, match="unique"):
        genetic_algorithm_optimize(
            model,
            length=6,
            initial=["ATGTAA"],
            objective=["maximize"],
            num_candidates=4,
            is_CDS_optimized=True,
            diversity=0,
        )
