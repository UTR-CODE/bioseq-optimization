import itertools

import numpy as np
import pytest

from sequence_optimization.motif_diversity import (
    pwm_similarity_matrix,
    soft_jaccard_distance_matrix,
    soft_position_distance_matrix,
    hierarchical_maxmin_selection,
)


def pwm(word):
    values=np.full((len(word),4),.01)
    for i,c in enumerate(word): values[i,'ACGT'.index(c)]=.97
    return values


def test_pwm_reverse_complement_and_informative_flanks_are_retained():
    a=pwm('ACGTCG');rc=a[::-1,::-1]
    s,_=pwm_similarity_matrix([a,rc,pwm('AAACGTCG'),pwm('CCACGTCG')])
    assert s[0,1]==pytest.approx(1)
    assert 0<s[0,2]<1 and 0<s[2,3]<1
    np.testing.assert_allclose(s,s.T)


def test_soft_jaccard_reduces_to_binary_and_softens_similar_motif_substitution():
    p=np.array([[1,0],[0,1],[1,1],[0,0],[0,0]])
    d,_=soft_jaccard_distance_matrix(p,np.eye(2))
    assert d[0,1]==1 and d[0,2]==.5 and d[3,4]==0 and d[0,3]==1
    s=np.array([[1,.8],[.8,1]])
    soft,fuzzy=soft_jaccard_distance_matrix(p,s)
    np.testing.assert_allclose(fuzzy[0],[1,.8])
    assert soft[0,1]==pytest.approx(.2) and 0<soft[0,2]<d[0,2]
    assert soft.min()>=0 and soft.max()<=1


def test_hungarian_matching_allows_similar_types_and_multiple_instances():
    instances=[[(0,0),(1,100)],[(1,1),(0,99)]]
    hard,count=soft_position_distance_matrix(instances,np.eye(2))
    assert hard[0,1]==99 and count[0,1]==2
    soft,_=soft_position_distance_matrix(instances,[[1,.8],[.8,1]])
    assert soft[0,1]==1
    a=[10,100];b=[11,40,95]
    d,n=soft_position_distance_matrix([[(0,x) for x in a],[(0,x) for x in b]],[[1]])
    brute=min(sum(abs(x-y) for x,y in zip(a,perm))/2 for perm in itertools.permutations(b,2))
    assert d[0,1]==brute and n[0,1]==2


def test_no_compatible_sites_have_no_positional_diversity_evidence():
    d,n=soft_position_distance_matrix([[(0,1)],[(1,199)],[]],np.eye(2))
    assert d[0,1]==d[0,2]==0 and n[0,1]==n[0,2]==0


def test_composition_priority_cannot_be_overridden_by_position_or_quality():
    c=np.array([[0,1,.9],[1,0,.5],[.9,.5,0]])
    p=np.array([[0,1,100],[1,0,99],[100,99,0]])
    chosen,_=hierarchical_maxmin_selection(c,p,[100,21,90],top_n=2)
    assert chosen==[0,1]


def test_position_then_quality_near_ties_are_lexicographic():
    c=np.array([[0,1,.99],[1,0,.5],[.99,.5,0]])
    p=np.array([[0,1,100],[1,0,99],[100,99,0]])
    chosen,trace=hierarchical_maxmin_selection(c,p,[100,90,80],top_n=2)
    assert chosen==[0,2] and trace[1]['composition_tie_candidates']==2
    p=np.array([[0,99,100],[99,0,1],[100,1,0]])
    assert hierarchical_maxmin_selection(c,p,[100,90,80],top_n=2)[0]==[0,1]


def test_maxmin_uses_all_selected_sequences_and_quality_gate():
    c=np.array([[0,1,.95,.6],[1,0,.05,.6],[.95,.05,0,.55],[.6,.6,.55,0]])
    p=np.zeros_like(c)
    assert hierarchical_maxmin_selection(c,p,[100,90,80,70],composition_tie=0)[0]==[0,1,3,2]
    chosen,_=hierarchical_maxmin_selection(c,p,[10,25,24,15],top_n=30,min_specificity=20)
    assert chosen==[1,2]
    assert hierarchical_maxmin_selection(c,p,[10,15,14,15],min_specificity=20)==([],[])


def test_bad_inputs_fail_before_selection():
    with pytest.raises(ValueError): soft_jaccard_distance_matrix([[2]],[[1]])
    with pytest.raises(ValueError): soft_jaccard_distance_matrix([[1]],[[.5]])
    with pytest.raises(ValueError): hierarchical_maxmin_selection([[0]],[[0]],[30],top_n=0)
    with pytest.raises(ValueError): soft_position_distance_matrix([[(1,10)]],[[1]])
