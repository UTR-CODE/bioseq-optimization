"""Variable-length scanning, independent-group validation and planted recovery."""
from collections import Counter
import numpy as np
import pandas as pd
import pytest
from sequence_optimization.simple_streme import (
    dinucleotide_controls, discover_seqlet_motifs, encode_fragments,
    optimal_threshold, paired_logp, scan_best,
)


def planted(seed=3):
    rng=np.random.default_rng(seed)
    sequences=[]
    motif='GATAAGCC'
    for _ in range(200):
        length=int(rng.integers(18,31))
        sequence=''.join(rng.choice(list('ACGT'),length))
        start=int(rng.integers(0,length-len(motif)+1))
        sequences.append(sequence[:start]+motif+sequence[start+len(motif):])
    return pd.DataFrame(dict(dna=sequences,sequence_id=[f'parent{i}' for i in range(200)]))


def test_shuffle_preserves_dinucleotides_and_lengths():
    sequences=['ACGTGATAAGCC', 'AAAAAAAAAA', 'ATATATCGCG', 'G']
    shuffled=dinucleotide_controls(sequences,7)
    assert shuffled==dinucleotide_controls(sequences,7)
    for a,b in zip(sequences,shuffled):
        assert len(a)==len(b)
        assert Counter(zip(a,a[1:]))==Counter(zip(b,b[1:]))


def test_variable_length_scanner_matches_both_strands_and_skips_padding():
    pwm=np.full((6,4),.01)
    pwm[np.arange(6),[0,1,2,3,0,0]]=.97
    codes=encode_fragments(['CCACGTAACC','TTACGT','ACG'])
    scores,starts,strands=scan_best(codes,pwm,np.full(4,.25))
    assert starts[0]==2 and strands[0]==1
    assert starts[1]==0 and strands[1]==-1
    assert scores[0]==pytest.approx(scores[1])
    assert scores[2]==-np.inf and starts[2]==-1


def test_training_threshold_agrees_with_exhaustive_paired_test():
    rng=np.random.default_rng(5)
    a=rng.normal(1,2,100);b=rng.normal(0,2,100)
    threshold,logp=optimal_threshold(a,b)
    possible=[(paired_logp(a,b,t),-t,t) for t in np.unique(a) if (a>=t).sum()>=2]
    expected=min(possible)
    assert threshold==expected[2] and logp==pytest.approx(expected[0])


def test_planted_motif_recovered_and_parent_split_never_leaks():
    frame=planted()
    result=discover_seqlet_motifs(frame,min_width=8,max_width=8,max_motifs=3)
    assert result.support.retained.any()
    passing=result.support[result.support.retained]
    assert (passing.seqlets>=10).all() and (passing.parents>=3).all()
    assert (passing.heldout_p_adjusted<=.05).all()
    pwm=result.pwms[passing.motif_id.iloc[0]]
    consensus=''.join(np.array(list('ACGT'))[pwm.argmax(1)])
    assert consensus in {'GATAAGCC','GGCTTATC'}
    assert result.split.groupby('sequence_id').split.nunique().eq(1).all()


def test_validation_sequences_cannot_change_discovered_pwms_or_thresholds():
    frame=planted()
    mask=np.arange(len(frame))%5==0
    controls=dinucleotide_controls(frame.dna,42)
    a=discover_seqlet_motifs(frame,controls=controls,validation_mask=mask,
        min_width=8,max_width=8,max_motifs=2)
    changed=frame.copy()
    changed.loc[mask,'dna']=np.asarray(controls)[mask]
    b=discover_seqlet_motifs(changed,controls=controls,validation_mask=mask,
        min_width=8,max_width=8,max_motifs=2)
    assert a.pwms.keys()==b.pwms.keys()
    for motif in a.pwms:np.testing.assert_array_equal(a.pwms[motif],b.pwms[motif])
    np.testing.assert_array_equal(a.support.threshold_bits,b.support.threshold_bits)


def test_reject_parent_split_between_training_and_validation():
    frame=planted()
    frame.loc[1,'sequence_id']=frame.loc[0,'sequence_id']
    with pytest.raises(ValueError,match='must not cross'):
        discover_seqlet_motifs(frame,validation_mask=np.arange(len(frame))%2==0)
