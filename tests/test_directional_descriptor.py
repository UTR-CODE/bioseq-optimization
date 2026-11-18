"""Directional RNA motifs must not acquire reverse-complement hits or proposals."""

import numpy as np
import pytest

from sequence_optimization import (
    CallableScorer,
    DiversitySearchConfig,
    MotifPositionDescriptor,
    random_seed_sequences,
)


from sequence_optimization.exploration import generate_diverse_designs


def test_directional_hits_and_ownership_do_not_merge_reverse_complements():
    descriptor = MotifPositionDescriptor(
        {"forward": ["AC"], "reverse": ["GT"]}, n_bins=1, reverse_complements=False
    )
    assert descriptor(["ACAA", "GTAA"]).tolist() == [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match="different motifs"):
        MotifPositionDescriptor({"a": ["AC"], "b": ["AC"]}, reverse_complements=False)
    with pytest.raises(ValueError, match="requires"):
        MotifPositionDescriptor(
            {"a": ["AC"]}, reverse_complements=False, include_strand=True
        )


def test_directional_proposals_insert_only_specified_orientation():
    descriptor = MotifPositionDescriptor(
        {"rna": ["GT"]}, n_bins=1, reverse_complements=False
    )
    values = descriptor.propose(["AAAAAAAA"] * 100, [0], np.random.default_rng(2))
    assert all("GT" in s and "AC" not in s for s in values)
    assert (descriptor(values)[:, 0] == 1).all()


def test_directionality_survives_search_metadata_round_trip():
    descriptor = MotifPositionDescriptor(
        {"rna": ["GT"]}, n_bins=2, reverse_complements=False
    )
    result = generate_diverse_designs(
        CallableScorer(lambda seqs: np.array([s.count("G") for s in seqs])),
        12,
        random_seed_sequences(12, 4, seed=1),
        ["maximize"],
        descriptor=descriptor,
        config=DiversitySearchConfig(
            total_budget=100, segment_budget=40, quality_fraction=0
        ),
        algorithms=["genetic_algorithm"],
        num_candidates=4,
        seed=1,
    )
    restored = MotifPositionDescriptor(**result.metadata["descriptor"])
    assert restored.reverse_complements is False
    np.testing.assert_array_equal(
        restored(["GTAAAAAAAAAA", "ACAAAAAAAAAA"]),
        descriptor(["GTAAAAAAAAAA", "ACAAAAAAAAAA"]),
    )
