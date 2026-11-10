"""Exact motif-presence classes and position-diverse sequence selection.

Presence is categorical. Distances are computed only within an identical
presence class; no sequence clustering or attribution weighting is involved.
"""

from collections.abc import Mapping, Sequence

import numpy as np


def presence_classes(presence):
    """Return deterministic labels and binary signatures, including all-zero.

    Labels depend only on the observed binary patterns, not input row order.
    The all-zero pattern is C000; nonempty patterns are numbered lexically.
    """
    values = np.asarray(presence)
    if values.ndim != 2 or not np.isin(values, [0, 1]).all():
        raise ValueError("presence must be a two-dimensional binary matrix")
    signatures = np.array(["".join(map(str, row.astype(int))) for row in values])
    nonzero = sorted(set(signatures) - {"0" * values.shape[1]})
    names = {key: f"C{i + 1:03d}" for i, key in enumerate(nonzero)}
    names["0" * values.shape[1]] = "C000"
    return np.array([names[key] for key in signatures]), signatures


def hungarian_position_distance(left: Mapping, right: Mapping):
    """Mean matched site displacement (bp), with equal weight per motif.

    Each mapping holds motif ID -> site centers in the same sequence frame.
    Rectangular assignment matches min(n_left, n_right) sites, without dummy
    sites or unmatched-site penalties. Motif presence must agree exactly.
    Strand, motif counts, model scores, and attribution are not distance terms.
    This permissive dissimilarity need not satisfy the triangle inequality.
    """
    from scipy.optimize import linear_sum_assignment

    def prepare(value):
        result = {}
        for motif, positions in value.items():
            sites = np.asarray(positions, dtype=float)
            if sites.ndim != 1 or not np.isfinite(sites).all():
                raise ValueError("site positions must be finite one-dimensional arrays")
            if len(sites):
                result[motif] = sites
        return result

    a, b = prepare(left), prepare(right)
    if set(a) != set(b):
        raise ValueError("position distance requires identical motif presence")
    if not a:
        return 0.0
    distances = []
    for motif in sorted(a):
        costs = np.abs(a[motif][:, None] - b[motif][None, :])
        rows, columns = linear_sum_assignment(costs)
        distances.append(float(costs[rows, columns].mean()))
    return float(np.mean(distances))


def maxmin_selection(
    distance,
    specificity,
    n_select=3,
    *,
    distance_tie_bp=2.0,
    min_distance_bp=0.0,
    candidate_ids: Sequence | None = None,
):
    """Greedy max-min selection, anchored at the highest specificity.

    Among candidates within distance_tie_bp of the best remaining minimum
    distance, choose the highest specificity, then the lexical candidate ID.
    A requested hard minimum applies to every selected pair. Return row
    indices and a trace exposing the best and achieved minimum at each step.
    A zero distance can be selected when min_distance_bp is zero; it is not
    evidence of position diversity. Exact duplicate DNAs should be collapsed
    by the caller before invoking this function.
    """
    d = np.asarray(distance, dtype=float)
    q = np.asarray(specificity, dtype=float)
    if q.ndim != 1 or not np.isfinite(q).all():
        raise ValueError("specificity must be a finite one-dimensional array")
    if d.shape != (len(q), len(q)) or not np.isfinite(d).all():
        raise ValueError("distance must be a finite square matrix matching specificity")
    if (d < 0).any() or not np.allclose(d, d.T) or not np.allclose(np.diag(d), 0):
        raise ValueError("distance must be nonnegative, symmetric, with zero diagonal")
    if isinstance(n_select, (bool, np.bool_)) or not isinstance(n_select, (int, np.integer)) or n_select < 1:
        raise ValueError("n_select must be a positive integer")
    for name, value in [("distance_tie_bp", distance_tie_bp), ("min_distance_bp", min_distance_bp)]:
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    ids = [str(i) for i in range(len(q))] if candidate_ids is None else list(map(str, candidate_ids))
    if len(ids) != len(q) or len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique and match specificity")
    if not len(q):
        return [], []
    first = min(range(len(q)), key=lambda i: (-q[i], ids[i]))
    chosen = [first]
    trace = [{"rank": 1, "index": first, "specificity": float(q[first]),
              "best_available_min_distance_bp": None, "min_distance_to_selected_bp": None,
              "distance_tie_candidates": 1, "selection_reason": "highest_specificity_anchor"}]
    while len(chosen) < min(n_select, len(q)):
        remaining = [i for i in range(len(q)) if i not in chosen]
        nearest = np.min(d[np.ix_(remaining, chosen)], axis=1)
        eligible = nearest >= min_distance_bp
        if not eligible.any():
            break
        best = float(nearest[eligible].max())
        close = [i for i, value, ok in zip(remaining, nearest, eligible)
                 if ok and value >= best - distance_tie_bp - 1e-12]
        pick = min(close, key=lambda i: (-q[i], ids[i]))
        achieved = float(d[pick, chosen].min())
        trace.append({"rank": len(chosen) + 1, "index": pick, "specificity": float(q[pick]),
                      "best_available_min_distance_bp": best,
                      "min_distance_to_selected_bp": achieved,
                      "distance_tie_candidates": len(close),
                      "selection_reason": "maxmin_position_then_specificity"})
        chosen.append(pick)
    return chosen, trace
