"""Evidence-aware hierarchy for motif-supported panels; no inferred zero distances."""
import numpy as np
from sequence_optimization.motif_diversity import hierarchical_maxmin_selection


def select_with_motif_evidence(composition, position, specificity, top_n, *,
                              matched_counts, presence, min_specificity=20.,
                              composition_tie=.02, position_tie_bp=2., candidate_ids=None,
                              require_motif_evidence=False, ignore_unmatched_position_pairs=False):
    if not require_motif_evidence and not ignore_unmatched_position_pairs:
        return hierarchical_maxmin_selection(composition, position, specificity, top_n,
            min_specificity=min_specificity, composition_tie=composition_tie,
            position_tie_bp=position_tie_bp, candidate_ids=candidate_ids)
    # Delegate validation of distances, scores, IDs and tolerances to the core API.
    hierarchical_maxmin_selection(composition, position, specificity, 1,
        min_specificity=min_specificity, composition_tie=composition_tie,
        position_tie_bp=position_tie_bp, candidate_ids=candidate_ids)
    if isinstance(top_n,(bool,np.bool_)) or not isinstance(top_n,(int,np.integer)) or top_n<1:
        raise ValueError('top_n must be a positive integer')
    c,p,q=np.asarray(composition),np.asarray(position),np.asarray(specificity)
    matched=np.asarray(matched_counts)
    pres=np.asarray(presence)
    if matched.shape!=p.shape or not np.isfinite(matched).all() or (matched<0).any() or not np.array_equal(matched,matched.T):
        raise ValueError('matched_counts must be finite, nonnegative and symmetric')
    if pres.ndim!=2 or len(pres)!=len(q) or not np.isin(pres,[0,1]).all():
        raise ValueError('presence must be binary with one row per sequence')
    ids=list(map(str,range(len(q)))) if candidate_ids is None else list(map(str,candidate_ids))
    eligible=q>=min_specificity
    if require_motif_evidence:eligible &= pres.any(1)
    pool=np.flatnonzero(eligible).tolist()
    if not pool:return [],[]
    def positional_nearest(i,chosen):
        others=[j for j in chosen if matched[i,j]>0] if ignore_unmatched_position_pairs else chosen
        # No evidence gives no position-diversity advantage.
        return float(p[i,others].min()) if others else 0.
    first=min(pool,key=lambda i:(-q[i],ids[i]));chosen=[first]
    trace=[dict(rank=1,index=first,specificity=float(q[first]),eligible_remaining=len(pool),
                best_composition_min=None,achieved_composition_min=None,composition_tie_candidates=None,
                best_position_min_bp=None,achieved_position_min_bp=None,position_tie_candidates=None,
                reason='highest_specificity_motif_supported_anchor')]
    while len(chosen)<min(top_n,len(pool)):
        remaining=[i for i in pool if i not in chosen]
        nearest={i:float(c[i,chosen].min()) for i in remaining};best_c=max(nearest.values())
        stage_c=[i for i in remaining if nearest[i]>=best_c-composition_tie-1e-12]
        pos={i:positional_nearest(i,chosen) for i in stage_c};best_p=max(pos.values())
        stage_p=[i for i in stage_c if pos[i]>=best_p-position_tie_bp-1e-12]
        pick=min(stage_p,key=lambda i:(-q[i],ids[i]))
        trace.append(dict(rank=len(chosen)+1,index=pick,specificity=float(q[pick]),eligible_remaining=len(remaining),
                          best_composition_min=best_c,achieved_composition_min=nearest[pick],composition_tie_candidates=len(stage_c),
                          best_position_min_bp=best_p,achieved_position_min_bp=pos[pick],position_tie_candidates=len(stage_p),
                          comparable_previously_selected=int((matched[pick,chosen]>0).sum()),
                          reason='composition_then_matched_position_then_specificity'))
        chosen.append(pick)
    return chosen,trace
