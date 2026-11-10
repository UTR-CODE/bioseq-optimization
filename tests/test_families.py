import numpy as np
import pandas as pd
import pytest

from sequence_optimization import kmer_position_features, select_diverse_families


def test_kmer_features_orientation_overlaps_and_alignment():
    f = kmer_position_features(["AAAACCCC", "GGGGTTTT"], ["AAA", "TTT"], n_bins=4)
    assert f.shape == (2, 4)
    assert f.iloc[0].tolist() == [1, 1, 0, 0]
    assert f.iloc[1].tolist() == [0, 0, 1, 1]
    np.testing.assert_array_equal(
        f.iloc[::-1],
        kmer_position_features(["GGGGTTTT", "AAAACCCC"], ["AAA"], n_bins=4),
    )


def data():
    d = pd.DataFrame(
        {
            "sequence": ["AAAA", "AAAC", "AACC", "CCCC", "CCCG", "CCGG"],
            "algorithm": ["a", "a", "b", "b", "b", "a"],
            "quality": [9, 8, 7, 6, 5, 4],
        }
    )
    return d, np.array([[4, 0]] * 3 + [[0, 4]] * 3)


@pytest.mark.parametrize("method", ["ward", "kmeans"])
def test_label_blind_family_balance_and_spacing(method):
    d, x = data()
    r = select_diverse_families(
        d,
        x,
        quality_column="quality",
        n_clusters=2,
        method=method,
        per_family=2,
        min_hamming=1,
    )
    assert r.recommendations.groupby("family").size().tolist() == [2, 2]
    assert set(r.recommendations.sequence) == {"AAAA", "AAAC", "CCCC", "CCCG"}
    assert r.family_summary.shortfall.sum() == 0
    changed = d.copy()
    changed["algorithm"] = ["changed"] * len(d)
    other = select_diverse_families(
        changed,
        x,
        quality_column="quality",
        n_clusters=2,
        method=method,
        per_family=2,
        min_hamming=1,
    )
    assert r.candidates.family.tolist() == other.candidates.family.tolist()
    assert (
        r.recommendations.sequence.tolist() == other.recommendations.sequence.tolist()
    )


def test_duplicate_provenance_and_unresolved(tmp_path):
    d, x = data()
    d = pd.concat(
        [
            d,
            pd.DataFrame(
                {
                    "sequence": ["AAAA", "TTTT"],
                    "algorithm": ["c", "c"],
                    "quality": [9, 10],
                }
            ),
        ],
        ignore_index=True,
    )
    x = np.vstack([x, [4, 0], [0, 0]])
    r = select_diverse_families(
        d, x, quality_column="quality", n_clusters=2, per_family=3, min_hamming=1
    )
    assert len(r.candidates) == 7
    assert len(r.input_assignments) == 8
    assert r.input_assignments.query("sequence == 'AAAA'").family.nunique() == 1
    assert r.candidates.query("sequence == 'TTTT'").family.iloc[0] == "Unresolved"
    assert "TTTT" not in set(r.recommendations.sequence)
    assert r.recommendations.sequence.is_unique
    assert r.algorithm_counts.loc["c"].sum() == 2
    r.save(tmp_path)
    assert (tmp_path / "recommendations.fa").read_text().count(">") == len(
        r.recommendations
    )
    assert len(pd.read_csv(tmp_path / "recommendations.csv")) == len(r.recommendations)


def test_insufficient_quality_returns_explicit_shortfall():
    d, x = data()
    r = select_diverse_families(
        d, x, quality_column="quality", n_clusters=2, min_quality=100
    )
    assert r.recommendations.empty
    assert r.family_summary.shortfall.sum() == 10


def test_minimum_distance_is_global():
    d, x = data()
    r = select_diverse_families(
        d,
        x,
        quality_column="quality",
        n_clusters=2,
        per_family=3,
        min_hamming=4,
        quality_quantile=0,
    )
    seqs = r.recommendations.sequence.tolist()
    assert all(
        sum(a != b for a, b in zip(s, t)) >= 4
        for i, s in enumerate(seqs)
        for t in seqs[i + 1 :]
    )
    assert r.family_summary.shortfall.sum() > 0


def test_conflicting_duplicate_vectors_rejected():
    d, x = data()
    d.loc[1, "sequence"] = "AAAA"
    x[1] = [0, 4]
    with pytest.raises(ValueError, match="identical feature"):
        select_diverse_families(d, x, quality_column="quality", n_clusters=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_clusters": True},
        {"quality_quantile": 2},
        {"method": "unknown"},
        {"per_family": 0},
    ],
)
def test_invalid_parameters(kwargs):
    d, x = data()
    with pytest.raises(ValueError):
        select_diverse_families(d, x, quality_column="quality", **kwargs)
