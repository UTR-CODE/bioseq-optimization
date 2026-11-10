import itertools

import numpy as np
import pytest

from sequence_optimization.motif_presence import (
    hungarian_position_distance,
    maxmin_selection,
    presence_classes,
)


def test_exact_presence_classification_is_not_similarity_clustering():
    p = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0]])
    labels, signatures = presence_classes(p)
    assert labels.tolist() == ["C001", "C001", "C002", "C000"]
    assert signatures.tolist() == ["101", "101", "111", "000"]
    reordered, _ = presence_classes(p[::-1])
    np.testing.assert_array_equal(reordered[::-1], labels)
    with pytest.raises(ValueError):
        presence_classes([[2]])


def test_rectangular_hungarian_matches_brute_force_and_ignores_extra_sites():
    a, b = [10, 80], [9, 40, 100]
    expected = min(sum(abs(x - y) for x, y in zip(a, ys)) / len(a)
                   for ys in itertools.permutations(b, len(a)))
    assert hungarian_position_distance({"M": a}, {"M": b}) == expected
    assert hungarian_position_distance({"M": b}, {"M": a}) == expected
    assert hungarian_position_distance({"M": [10]}, {"M": [10, 150]}) == 0
    # Each motif has equal weight regardless of its number of sites.
    assert hungarian_position_distance({"M": [0, 80], "N": [50]},
                                       {"M": [10, 90], "N": [90]}) == 25
    assert hungarian_position_distance({}, {}) == 0
    with pytest.raises(ValueError):
        hungarian_position_distance({"M": [1]}, {"N": [1]})


def test_maxmin_uses_all_selected_not_only_the_anchor():
    # After selecting 0 and 1, point 2 is far from the anchor but close to 1.
    d = np.array([[0, 100, 95, 60], [100, 0, 5, 60],
                  [95, 5, 0, 55], [60, 60, 55, 0]], float)
    chosen, trace = maxmin_selection(d, [10, 9, 8, 7], distance_tie_bp=0)
    assert chosen == [0, 1, 3]
    assert trace[2]["min_distance_to_selected_bp"] == 60


def test_near_distance_tie_prefers_specificity_and_respects_hard_minimum():
    d = np.array([[0, 20, 19], [20, 0, 1], [19, 1, 0]], float)
    chosen, trace = maxmin_selection(d, [100, 1, 10], n_select=2, distance_tie_bp=2)
    assert chosen == [0, 2]
    assert trace[1]["best_available_min_distance_bp"] == 20
    assert trace[1]["min_distance_to_selected_bp"] == 19
    chosen, _ = maxmin_selection(d, [100, 1, 10], n_select=3,
                                distance_tie_bp=2, min_distance_bp=19.5)
    assert chosen == [0, 1]


def test_no_sites_and_small_classes_do_not_invent_diversity():
    chosen, trace = maxmin_selection(np.zeros((2, 2)), [2, 1], n_select=3)
    assert chosen == [0, 1]
    assert trace[1]["min_distance_to_selected_bp"] == 0
    assert maxmin_selection(np.zeros((0, 0)), [], n_select=3) == ([], [])
    assert maxmin_selection(np.zeros((1, 1)), [3])[0] == [0]


@pytest.mark.parametrize("kwargs", [{"n_select": 0}, {"n_select": 1.5},
                                    {"distance_tie_bp": -1}, {"min_distance_bp": np.nan}])
def test_invalid_selection_options(kwargs):
    with pytest.raises(ValueError):
        maxmin_selection(np.zeros((1, 1)), [1], **kwargs)
