import numpy as np
from sequence_optimization.motif_evidence import select_with_motif_evidence


def test_no_match_does_not_flatten_other_position_evidence():
    composition=np.ones((4,4))-np.eye(4)
    position=np.array([[0,20,25,24],[20,0,0,5],[25,0,0,10],[24,5,10,0]],float)
    counts=np.ones((4,4),int)
    counts[1,2]=counts[2,1]=0
    options=dict(matched_counts=counts,presence=np.ones((4,1),int),min_specificity=20.,composition_tie=0.,position_tie_bp=0.)
    old,_=select_with_motif_evidence(composition,position,[30,29,28,27],3,**options)
    new,trace=select_with_motif_evidence(composition,position,[30,29,28,27],3,ignore_unmatched_position_pairs=True,**options)
    assert old==[0,2,3]
    assert new==[0,2,1]
    assert trace[-1]['comparable_previously_selected']==1
    assert trace[-1]['achieved_position_min_bp']==20.


def test_catalog_evidence_gate_excludes_unannotated_high_score():
    d=np.ones((3,3))-np.eye(3)
    selected,_=select_with_motif_evidence(d,d,[40,30,29],3,matched_counts=np.ones((3,3)),presence=np.array([[0],[1],[1]]),require_motif_evidence=True)
    assert selected==[1,2]


def test_no_position_evidence_confers_no_diversity_advantage():
    d=np.ones((3,3))-np.eye(3)
    selected,trace=select_with_motif_evidence(d,np.zeros((3,3)),[30,29,28],3,matched_counts=np.zeros((3,3)),presence=np.eye(3,dtype=int),ignore_unmatched_position_pairs=True)
    assert selected==[0,1,2]
    assert trace[-1]['achieved_position_min_bp']==0.
