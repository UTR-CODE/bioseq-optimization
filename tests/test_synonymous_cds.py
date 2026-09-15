import numpy as np
import pytest

from sequence_optimization import (
    CallableScorer,
    basesearch_optimize,
    genetic_algorithm_optimize,
)
from sequence_optimization.codons import GENETIC_CODE, STOP_CODONS, SynonymousCodonSpace


def test_synonymous_saturation_and_invalid_seeds():
    from sequence_optimization import SaturationConfig

    reference = "ATG" + "GCT" * 4 + "TAA"
    space = SynonymousCodonSpace(reference, {"A": ["GCC", "GCG"]})

    def score(values):
        space.validate(values)
        return np.array([sequence.count("G") for sequence in values])

    scorer = CallableScorer(score)
    result = genetic_algorithm_optimize(
        scorer,
        length=len(reference),
        reference_cds=reference,
        synonymous_codons={"A": ["GCC", "GCG"]},
        initial=[
            "ATG" + "GCC" * 4 + "TAA",
            "ATG" + "GCG" + "GCC" * 3 + "TAA",
            "ATG" + "GCG" * 2 + "GCC" * 2 + "TAA",
            "ATG" + "GCG" * 3 + "GCC" + "TAA",
        ],
        num_candidates=4,
        diversity=0,
        saturation=SaturationConfig(
            minimum_queries=4, plateau_patience_queries=4, maximum_queries=12
        ),
    )
    for selection in result.selections.values():
        space.validate(selection.sequences)
    with pytest.raises(ValueError, match="allowed codons"):
        genetic_algorithm_optimize(
            scorer,
            length=len(reference),
            reference_cds=reference,
            synonymous_codons={"A": ["GCC", "GCG"]},
            initial=[reference],
            num_candidates=1,
            budget=4,
        )


def test_synonymous_search_only_scores_allowed_cds():
    reference = "ATG" + "GCT" * 8 + "TAA"
    allowed = {"A": ["GCC", "GCG"]}
    space = SynonymousCodonSpace(reference, allowed)
    queried = []

    def score(values):
        space.validate(values)
        queried.extend(values)
        return np.array([value.count("G") for value in values])

    result = genetic_algorithm_optimize(
        CallableScorer(score),
        length=len(reference),
        reference_cds=reference,
        synonymous_codons=allowed,
        initial=[
            "ATG" + "GCC" * 8 + "TAA",
            "ATG" + "GCC" + "GCG" + "GCC" * 6 + "TAA",
            "ATG" + "GCC" + "GCG" * 2 + "GCC" * 5 + "TAA",
            "ATG" + "GCC" + "GCG" * 3 + "GCC" * 4 + "TAA",
        ],
        fixed_region=["5:6"],
        num_candidates=4,
        diversity=0,
        budget=40,
        mutation_rate=0.3,
        seed=7,
    )
    assert len(queried) > 4
    assert result.query_count <= 40
    assert len(set(result.sequences)) == 4
    for value in queried:
        assert value[5] == "C"
        assert value[:3] == "ATG" and value[-3:] in STOP_CODONS
        assert (
            "".join(GENETIC_CODE[value[i : i + 3]] for i in range(0, len(value), 3))
            == "M" + "A" * 8 + "*"
        )


@pytest.mark.parametrize("stop_codon", sorted(STOP_CODONS))
def test_all_standard_terminal_stop_codons_are_accepted(stop_codon):
    reference = "ATGGCT" + stop_codon
    space = SynonymousCodonSpace(reference)
    candidates = ["ATGGCTTAA", "ATGGCCTAG", "ATGGCATGA"]
    assert space.validate(candidates).tolist() == candidates


def test_cds_candidate_validation_checks_boundaries_and_translation():
    space = SynonymousCodonSpace("ATGGCTTAA")
    with pytest.raises(ValueError, match="begin with ATG"):
        space.validate(["GTGGCTTAA"])
    with pytest.raises(ValueError, match="end with TAA, TAG, or TGA"):
        space.validate(["ATGGCTAAA"])
    with pytest.raises(ValueError, match="preserve the reference protein"):
        space.validate(["ATGGATTAA"])


@pytest.mark.parametrize(
    "reference,table",
    [
        ("ATGGCT", None),
        ("GTGGCTTAA", None),
        ("ATGGCTAAA", None),
        ("ATGTAGGCTTAA", None),
        ("ATGGCTTAA", {"A": ["GAA"]}),
        ("ATGGCTTAA", {"A": []}),
    ],
)
def test_invalid_constraints(reference, table):
    with pytest.raises(ValueError):
        SynonymousCodonSpace(reference, table)


def test_capacity_and_unsupported_algorithm():
    scorer = CallableScorer(lambda values: np.zeros(len(values)))
    with pytest.raises(ValueError, match="unique"):
        genetic_algorithm_optimize(
            scorer,
            length=6,
            reference_cds="ATGTAA",
            synonymous_codons={"*": ["TAA"]},
            num_candidates=2,
            budget=4,
        )
    with pytest.raises(ValueError, match="genetic_algorithm"):
        basesearch_optimize(
            scorer, length=9, reference_cds="ATGGCTTAA", num_candidates=2, budget=4
        )
    with pytest.raises(ValueError, match="reference_cds"):
        genetic_algorithm_optimize(
            scorer,
            length=9,
            synonymous_codons={"A": ["GCT"]},
            num_candidates=2,
            budget=4,
        )
