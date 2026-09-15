import numpy as np
import pytest

from sequence_optimization import (
    CallableScorer,
    SaturationConfig,
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
            minimum_queries=4, plateau_patience_queries=4, maximum_queries=12
        ),
    )
    for selection in result.selections.values():
        space.validate(selection.sequences)


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
        budget=8,
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
        budget=40,
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


def test_cds_optimizer_validation_and_supported_algorithm():
    model = CallableScorer(lambda values: np.zeros(len(values)))
    common = dict(objective=["maximize"], num_candidates=2, budget=4)
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
    with pytest.raises(ValueError, match="genetic_algorithm"):
        basesearch_optimize(
            model,
            length=9,
            initial=["ATGGCTTAA"],
            is_CDS_optimized=True,
            **common,
        )
    with pytest.raises(ValueError, match="complete ACGT CDS in frame"):
        genetic_algorithm_optimize(
            model,
            length=8,
            initial=["ATGGCTAA"],
            is_CDS_optimized=True,
            **common,
        )


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
            budget=4,
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
            budget=4,
        )
