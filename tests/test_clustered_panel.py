"""End-to-end candidate clustering, highlighting and final-panel checks."""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from sequence_optimization import (
    DiversitySearchResult,
    MotifPositionDescriptor,
    cluster_designs,
)


def fixture_pool():
    rng = np.random.default_rng(7)
    sequences = []
    for position in (0, 8, 16):
        for _ in range(12):
            seq = "".join(rng.choice(list("AT"), size=24))
            sequences.append(seq[:position] + "GGCC" + seq[position + 4 :])
    return pd.DataFrame(
        dict(
            sequence=sequences,
            algorithm=["a", "b", "c"] * 12,
            quality=np.tile(np.arange(12), 3) + 5.0,
        )
    )


def descriptor():
    return MotifPositionDescriptor({"GC": ["GGCC"]}, n_bins=3)


def panel(**kwargs):
    options = dict(
        descriptor=descriptor(),
        n_clusters=3,
        per_family=3,
        final_count=5,
        min_quality=8,
        min_hamming=2,
        stability_repeats=3,
    )
    options.update(kwargs)
    return cluster_designs(fixture_pool(), **options)


def test_groups_highlights_and_small_panel_keep_quality_and_global_spacing():
    result = panel(include_sequence_features=False)
    assert result.candidates.family.nunique() == 3
    assert len(result.representatives) == 9
    assert len(result.recommendations) == 5
    assert result.recommendations.family.nunique() == 3
    assert result.representatives.quality.ge(8).all()
    assert set(result.recommendations.sequence) <= set(result.representatives.sequence)
    assert {"central", "quality_lead", "diverse"} == set(
        result.representatives.representative_role
    )
    assert result.candidates.final_rank.gt(0).sum() == 5
    for a, b in itertools.combinations(result.recommendations.sequence, 2):
        assert sum(x != y for x, y in zip(a, b)) >= 2
    assert result.feature_profiles.feature.str.startswith("motif:").all()


def test_signed_extra_features_are_standardized_and_units_do_not_change_partition():
    data = fixture_pool()
    extra = pd.DataFrame(
        {"structure": np.repeat([-8.0, 0.0, 8.0], 12)}, index=data.index
    )
    options = dict(
        descriptor=descriptor(),
        n_clusters=3,
        feature_blocks={"structure": extra},
        block_weights={"structure": 0.5},
        stability_repeats=0,
        min_hamming=1,
    )
    result = cluster_designs(data, **options)
    options["feature_blocks"] = {"structure": extra * 10000 + 76}
    rescaled = cluster_designs(data, **options)
    assert result.candidates.family.tolist() == rescaled.candidates.family.tolist()
    assert result.metadata["feature_blocks"]["structure"]["retained_columns"] == 1


def test_input_order_and_algorithm_labels_do_not_define_clusters():
    data = fixture_pool()
    base = cluster_designs(
        data, descriptor=descriptor(), n_clusters=3, min_hamming=1, stability_repeats=0
    )
    shuffled = data.sample(frac=1, random_state=44)
    shuffled["algorithm"] = "unrelated_label"
    other = cluster_designs(
        shuffled,
        descriptor=descriptor(),
        n_clusters=3,
        min_hamming=1,
        stability_repeats=0,
    )
    assert base.candidates.sequence.tolist() == other.candidates.sequence.tolist()
    assert base.candidates.family.tolist() == other.candidates.family.tolist()
    assert (
        base.recommendations.sequence.tolist()
        == other.recommendations.sequence.tolist()
    )


def test_duplicate_provenance_preserved_and_extra_features_must_agree():
    data = fixture_pool()
    duplicate = data.iloc[:1].assign(algorithm="other", quality=100.0)
    data = pd.concat([data, duplicate], ignore_index=True)
    result = cluster_designs(
        data, descriptor=descriptor(), n_clusters=3, stability_repeats=0
    )
    assert len(result.candidates) == 36
    assert len(result.input_assignments) == 37
    row = result.candidates[
        result.candidates.sequence.eq(duplicate.sequence.iloc[0])
    ].iloc[0]
    assert row.quality == 100
    assert json.loads(row.source_algorithms) == ["a", "other"]
    extra = np.zeros((37, 1))
    extra[-1] = 1
    with pytest.raises(ValueError, match="inconsistent"):
        cluster_designs(data, descriptor=descriptor(), feature_blocks={"x": extra})


def test_shortfall_does_not_relax_quality_and_empty_report_still_saves(tmp_path):
    result = panel(min_quality=100.0)
    assert result.representatives.empty
    assert result.recommendations.empty
    assert result.metadata["shortfall"] == 5
    result.save(tmp_path)
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "recommendations.fa").read_text() == ""
    assert pd.read_csv(tmp_path / "recommendations.csv").empty


def test_automatic_cluster_count_and_report_exports(tmp_path):
    pytest.importorskip("matplotlib")
    result = panel(n_clusters=None, include_sequence_features=False)
    assert result.metadata["n_clusters"] == 3
    assert result.diagnostics.selected.sum() == 1
    result.save(tmp_path)
    assert (tmp_path / "clusters.svg").exists()
    assert (tmp_path / "clusters.pdf").exists()
    report = (tmp_path / "report.html").read_text()
    assert "data-family=" in report and "SEQ" in report
    assert len((tmp_path / "recommendations.fa").read_text().split(">")) - 1 == 5


def test_result_shortlist_inherits_search_thresholds_and_supports_larger_archive():
    data = fixture_pool()
    search = DiversitySearchResult(
        data.iloc[:20].copy(),
        data,
        pd.DataFrame(),
        dict(
            descriptor=dict(motifs={"GC": ["GGCC"]}, n_bins=3),
            quality_floor=12.0,
            diversity=2,
        ),
    )
    result = search.shortlist(
        source="archive", n_clusters=3, final_count=4, stability_repeats=0
    )
    assert len(result.candidates) == 36
    assert result.metadata["min_quality"] == 12
    assert result.metadata["min_hamming"] == 2
    assert result.recommendations.quality.ge(12).all()


def test_feature_index_alignment_and_nonfinite_inputs_are_rejected():
    data = fixture_pool()
    with pytest.raises(ValueError, match="index"):
        cluster_designs(
            data,
            descriptor=descriptor(),
            feature_blocks={"x": pd.DataFrame({"x": range(36)}, index=range(1, 37))},
        )
    with pytest.raises(ValueError, match="finite"):
        cluster_designs(
            data,
            descriptor=descriptor(),
            feature_blocks={"x": np.full((36, 1), np.nan)},
        )


def test_one_sequence_and_no_motif_are_reported_without_inventing_clusters():
    data = pd.DataFrame(dict(sequence=["AAAAAAAA"], quality=[10.0]))
    result = cluster_designs(data, descriptor=descriptor(), min_hamming=0)
    assert result.metadata["n_clusters"] == 1
    assert result.recommendations.empty
    assert not result.candidates.motif_detected.any()
    result = cluster_designs(
        data, descriptor=descriptor(), min_hamming=0, require_motif=False
    )
    assert len(result.recommendations) == 1
    assert result.metadata["shortfall"] == 5
