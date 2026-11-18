"""Soft motif-composition distances and lexicographic diversity selection.

PWM side columns are retained. Motif composition uses binary presence, while
positional matching allows repeated motif instances and ignores unmatched sites.
"""

import numpy as np


def _similarity_matrix(values):
    s = np.asarray(values, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not np.isfinite(s).all():
        raise ValueError("motif similarity must be a finite square matrix")
    if (s < 0).any() or (s > 1).any() or not np.allclose(s, s.T) or not np.allclose(np.diag(s), 1):
        raise ValueError("motif similarity must be symmetric, in [0,1], with unit diagonal")
    return s


def pwm_similarity_matrix(pwms, min_overlap=6, *, reverse_complement=True):
    """Best strand/shift cosine of full-span information-weighted PWM contrast.

    For each PWM, X=(P-0.25)*sqrt(2-H(P)). Background padding has zero contrast.
    Every column enters the norm, including unaligned informative flanks.
    Shifts are ungapped, with at least min_overlap columns shared (or the full
    shorter width when it is shorter). Negative cosine values become zero.
    No contribution weights, motif trimming, or TF-family merging is used.
    Set reverse_complement=False for supplied-orientation-only comparisons.
    Return similarities and upper-triangle alignment metadata.
    """
    if isinstance(min_overlap, bool) or not isinstance(min_overlap, (int, np.integer)) or min_overlap < 1:
        raise ValueError("min_overlap must be a positive integer")
    if not isinstance(reverse_complement, (bool, np.bool_)):
        raise ValueError('reverse_complement must be a boolean')
    contrasts = []
    for pwm in pwms:
        p = np.asarray(pwm, dtype=float)
        if p.ndim != 2 or p.shape[1] != 4 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
            raise ValueError("PWMs must have shape (width,4) with probability rows")
        information = np.maximum(0, 2 + (p * np.log2(np.maximum(p, 1e-300))).sum(1))
        contrasts.append((p - .25) * np.sqrt(information[:, None]))
    result = np.eye(len(contrasts)); alignments = []
    for i, a in enumerate(contrasts):
        for j in range(i):
            b0 = contrasts[j]; norm = np.linalg.norm(a) * np.linalg.norm(b0)
            required = min(min_overlap, len(a), len(b0))
            best = (0., 0, "+", 0)
            if norm > 0:
                orientations = [("+", b0), ("-", b0[::-1, ::-1])] if reverse_complement else [("+", b0)]
                for strand, b in orientations:
                    for offset in range(-len(b) + required, len(a) - required + 1):
                        lo, hi = max(0, offset), min(len(a), offset + len(b))
                        score = float(np.sum(a[lo:hi] * b[lo-offset:hi-offset]) / norm)
                        if score > best[0]:
                            best = (score, offset, strand, hi - lo)
            elif np.linalg.norm(a) == np.linalg.norm(b0) == 0:
                best = (1., 0, "+", min(len(a), len(b0)))
            result[i, j] = result[j, i] = np.clip(best[0], 0, 1)
            alignments.append(dict(index_a=i, index_b=j, similarity=float(result[i,j]),
                                   b_start_relative_to_a=best[1], b_orientation=best[2], overlap=best[3]))
    return result, alignments


def soft_jaccard_distance_matrix(presence, motif_similarity):
    """Fuzzy-presence Jaccard: max propagation, then sum(min)/sum(max).

    F[i,k]=max_{m present in i} similarity[m,k]; empty presence maps to zeros.
    This retains flank-sensitive similarities without summing redundant motif
    variants. Identity motif similarity reduces exactly to binary Jaccard.
    Empty/empty distance is zero; empty/nonempty distance is one.
    Return the sequence distance matrix and fuzzy-presence features.
    """
    s = _similarity_matrix(motif_similarity); p = np.asarray(presence)
    if p.ndim != 2 or p.shape[1] != len(s) or not np.isin(p, [0, 1]).all():
        raise ValueError("presence must be binary with one column per motif")
    patterns, inverse = np.unique(p, axis=0, return_inverse=True)
    fuzzy = np.zeros(patterns.shape, dtype=float)
    for i, row in enumerate(patterns):
        if row.any(): fuzzy[i] = s[row.astype(bool)].max(0)
    d = np.zeros((len(patterns), len(patterns)), dtype=float)
    for i in range(len(patterns)):
        intersection = np.minimum(fuzzy[i], fuzzy).sum(1)
        union = np.maximum(fuzzy[i], fuzzy).sum(1)
        d[i] = 1 - np.divide(intersection, union, out=np.ones_like(union), where=union > 0)
    d = np.clip((d + d.T) / 2, 0, 1); np.fill_diagonal(d, 0)
    return d[np.ix_(inverse, inverse)], fuzzy[inverse]


def _match_positions(types_a, pos_a, types_b, pos_b, s, min_similarity):
    from scipy.optimize import linear_sum_assignment

    if not len(types_a) or not len(types_b): return 0., 0
    allowed = s[types_a[:, None], types_b[None, :]] >= min_similarity
    if not allowed.any(): return 0., 0
    shifts = np.abs(pos_a[:, None] - pos_b[None, :])
    # A forbidden-edge penalty dominates all legitimate shifts in an entire
    # assignment. Thus first maximize compatible cardinality, then minimize
    # total displacement. Forced forbidden pairs are dropped, never penalized
    # in the returned positional distance.
    penalty = (min(len(pos_a), len(pos_b)) + 1) * (float(shifts.max()) + 1)
    cost = np.where(allowed, shifts, penalty)
    rows, cols = linear_sum_assignment(cost); ok = allowed[rows, cols]
    rows, cols = rows[ok], cols[ok]
    return (float(shifts[rows, cols].mean()), len(rows)) if len(rows) else (0., 0)


def soft_position_distance_matrix(instances, motif_similarity, min_similarity=.5, progress=None):
    """Pairwise Hungarian position distances and compatible match counts.

    Each sequence contains [(motif_index, center_bp), ...]. Retain all repeated
    sites. Eligible motif pairs meet min_similarity. Match as many compatible
    sites as possible, then minimize total absolute displacement. The reported
    distance is mean bp displacement over matched pairs; unmatched sites do not
    enter a penalty. No compatible pairs returns zero AND count zero, denoting
    lack of positional evidence rather than large diversity.
    """
    s = _similarity_matrix(motif_similarity)
    if not np.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be in [0,1]")
    keys = {}; inverse = []
    for seq in instances:
        key = tuple(sorted((int(m), float(p)) for m, p in seq))
        if any(m < 0 or m >= len(s) or not np.isfinite(p) for m, p in key):
            raise ValueError("invalid motif index or position")
        inverse.append(keys.setdefault(key, len(keys)))
    packed = [(np.array([m for m,p in key], dtype=int), np.array([p for m,p in key])) for key in keys]
    d = np.zeros((len(keys), len(keys)), dtype=float); counts = np.zeros_like(d, dtype=np.int32)
    for i, (ta, pa) in enumerate(packed):
        counts[i,i] = len(ta)
        for j in range(i):
            tb, pb = packed[j]; value, n = _match_positions(ta, pa, tb, pb, s, min_similarity)
            d[i,j] = d[j,i] = value; counts[i,j] = counts[j,i] = n
        if progress is not None and (i % 100 == 0 or i == len(packed)-1): progress(i+1, len(packed))
    return d[np.ix_(inverse,inverse)], counts[np.ix_(inverse,inverse)]


def hierarchical_maxmin_selection(composition, position, specificity, top_n=30, *,
                                  min_specificity=20., composition_tie=.02,
                                  position_tie_bp=2., candidate_ids=None):
    """Quality gate -> composition max-min -> positional max-min -> quality.

    Highest specificity seeds the panel. At every subsequent step, use minimum
    distance to ALL already selected sequences at each diversity stage. Only
    candidates within composition_tie of the best composition minimum advance
    to positional selection; then position_tie_bp defines near ties resolved by
    specificity and stable ID. No weighted sum of the three objectives is used.
    Return original row indices plus an auditable selection trace.
    """
    q = np.asarray(specificity, dtype=float)
    if q.ndim != 1 or not np.isfinite(q).all(): raise ValueError("specificity must be finite and one-dimensional")
    c, p = np.asarray(composition, dtype=float), np.asarray(position, dtype=float)
    for name, d in [("composition", c), ("position", p)]:
        if d.shape != (len(q), len(q)) or not np.isfinite(d).all() or (d < 0).any() or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0):
            raise ValueError(f"{name} must be finite, symmetric, nonnegative with zero diagonal")
    if (c > 1+1e-12).any(): raise ValueError("composition distance must be in [0,1]")
    if isinstance(top_n,(bool,np.bool_)) or not isinstance(top_n,(int,np.integer)) or top_n < 1: raise ValueError("top_n must be positive integer")
    if not np.isfinite([min_specificity,composition_tie,position_tie_bp]).all() or composition_tie < 0 or position_tie_bp < 0: raise ValueError("invalid gate or diversity tolerance")
    ids = list(map(str,range(len(q)))) if candidate_ids is None else list(map(str,candidate_ids))
    if len(ids)!=len(q) or len(set(ids))!=len(ids): raise ValueError("candidate IDs must be unique")
    pool = np.flatnonzero(q >= min_specificity).tolist()
    if not pool: return [], []
    first = min(pool,key=lambda i:(-q[i],ids[i])); selected=[first]
    trace=[dict(rank=1,index=first,specificity=float(q[first]),eligible_remaining=len(pool),
                best_composition_min=None,achieved_composition_min=None,composition_tie_candidates=None,
                best_position_min_bp=None,achieved_position_min_bp=None,position_tie_candidates=None,
                reason="highest_specificity_anchor")]
    while len(selected) < min(top_n,len(pool)):
        remaining=[i for i in pool if i not in selected]
        near_c=c[np.ix_(remaining,selected)].min(1); best_c=float(near_c.max())
        c_stage=[i for i,value in zip(remaining,near_c) if value >= best_c-composition_tie-1e-12]
        near_p=p[np.ix_(c_stage,selected)].min(1); best_p=float(near_p.max())
        p_stage=[i for i,value in zip(c_stage,near_p) if value >= best_p-position_tie_bp-1e-12]
        pick=min(p_stage,key=lambda i:(-q[i],ids[i]))
        trace.append(dict(rank=len(selected)+1,index=pick,specificity=float(q[pick]),eligible_remaining=len(remaining),
                          best_composition_min=best_c,achieved_composition_min=float(c[pick,selected].min()),
                          composition_tie_candidates=len(c_stage),best_position_min_bp=best_p,
                          achieved_position_min_bp=float(p[pick,selected].min()),position_tie_candidates=len(p_stage),
                          reason="composition_then_position_then_specificity"))
        selected.append(pick)
    return selected, trace
