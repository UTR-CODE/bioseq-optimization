"""Shared FASTA, candidate provenance and sequence-panel operations."""

import hashlib
from pathlib import Path

import numpy as np


def write_fasta(path, ids, sequences):
    ids, sequences = list(ids), list(sequences)
    if len(ids) != len(sequences) or len(set(map(str, ids))) != len(ids):
        raise ValueError('FASTA IDs must be unique and match the sequence count')
    if any(any(c.isspace() for c in str(i)) for i in ids):
        raise ValueError('FASTA IDs cannot contain whitespace')
    Path(path).write_text(''.join(f'>{i}\n{s}\n' for i, s in zip(ids, sequences)))


def deduplicate_candidates(records):
    """Collapse exact DNA, preserving a separate many-to-one provenance table."""
    records = records.copy()
    records['sequence_id'] = records.sequence.map(
        lambda s: 'S_'+hashlib.sha256(s.encode()).hexdigest()[:20])
    if records.groupby('sequence_id').sequence.nunique().max() != 1:
        raise RuntimeError('Sequence ID collision')
    unique = records.sort_values(['score', 'record_id'], ascending=[False, True]).drop_duplicates('sequence').copy()
    sources = records.groupby('sequence_id').algorithm.agg(lambda v: ';'.join(sorted(set(v))))
    unique['algorithms'] = unique.sequence_id.map(sources)
    return unique.reset_index(drop=True), records


def select_sequence_panel(records, *, top_n=10, min_score, distance_tie_nt=1):
    """Fallback for tasks without a validated motif catalog: DNA Hamming Max-Min.

    This selects sequence diversity only and makes no claim of motif diversity.
    """
    unique,_=deduplicate_candidates(records)
    pool=unique[unique.score>=min_score].copy().reset_index(drop=True)
    if isinstance(top_n,bool) or not isinstance(top_n,int) or top_n<1 or distance_tie_nt<0:
        raise ValueError('Invalid panel size or distance tie tolerance')
    if len(pool)<top_n:raise ValueError('Too few unique sequences pass min_score')
    if pool.sequence.str.len().nunique()!=1:raise ValueError('Hamming requires equal lengths')
    codes=np.array([list(s) for s in pool.sequence]);distance=(codes[:,None,:]!=codes[None,:,:]).sum(2)
    chosen=[int(pool.score.to_numpy().argmax())]
    while len(chosen)<top_n:
        remaining=[i for i in range(len(pool)) if i not in chosen]
        nearest=distance[np.ix_(remaining,chosen)].min(1)
        best=nearest.max();stage=[i for i,v in zip(remaining,nearest) if v>=best-distance_tie_nt]
        chosen.append(min(stage,key=lambda i:(-pool.score.iloc[i],pool.sequence_id.iloc[i])))
    selected=pool.iloc[chosen].copy();selected.insert(0,'selection_rank',range(1,top_n+1))
    return selected,distance[np.ix_(chosen,chosen)]
