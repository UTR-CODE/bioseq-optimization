"""Portable MEME/FIMO I/O and motif-evidence-aware panel selection."""
from pathlib import Path
import json
import re
import subprocess
import numpy as np
import pandas as pd
from .workflows import deduplicate_candidates, write_fasta
from .motif_diversity import (pwm_similarity_matrix, soft_jaccard_distance_matrix,
                              soft_position_distance_matrix)
from .motif_evidence import select_with_motif_evidence

FIMO_COLUMNS = ['motif_id','sequence_name','start','stop','strand','p-value','q-value']


def read_meme(path):
    """Read DNA probability matrices; require explicit A/C/G/T alphabet."""
    lines=Path(path).read_text().splitlines()
    alphabet=[x.split('=',1)[1].strip().replace(' ','') for x in lines if x.startswith('ALPHABET=')]
    if alphabet != ['ACGT']:raise ValueError('Expected exactly ALPHABET= ACGT')
    motifs={};name=None
    for i,line in enumerate(lines):
        if line.startswith('MOTIF '):name=line.split()[1]
        if line.startswith('letter-probability matrix:'):
            match=re.search(r'\bw\s*=\s*(\d+)',line)
            if not match or name is None or name in motifs:raise ValueError('Invalid or duplicate motif matrix')
            width=int(match.group(1))
            p=np.array([[float(v) for v in row.split()] for row in lines[i+1:i+1+width]])
            if p.shape!=(width,4) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-4):
                raise ValueError(f'Invalid probability matrix: {name}')
            motifs[name]=p/p.sum(1,keepdims=True)
    if not motifs:raise ValueError('No motif matrices found')
    return motifs


def write_meme(path, motifs):
    lines=['MEME version 5','','ALPHABET= ACGT','','strands: + -','',
           'Background letter frequencies','A 0.25 C 0.25 G 0.25 T 0.25','']
    for name,pwm in motifs.items():
        p=np.asarray(pwm,float)
        if not name or any(c.isspace() for c in name) or p.ndim!=2 or p.shape[1]!=4 or not len(p) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):
            raise ValueError('Expected motif ID -> (width,4) probability matrix')
        lines.extend([f'MOTIF {name}',f'letter-probability matrix: alength= 4 w= {len(p)} nsites= 10',
                      *[' '.join(f'{v:.10f}' for v in row) for row in p],''])
    Path(path).write_text('\n'.join(lines)+'\n')


def read_fimo(path):
    try:d=pd.read_csv(path,sep='\t',comment='#')
    except pd.errors.EmptyDataError:return pd.DataFrame(columns=FIMO_COLUMNS)
    if not set(FIMO_COLUMNS)<=set(d):raise ValueError(f'FIMO table requires {FIMO_COLUMNS}')
    d=d.dropna(subset=['motif_id','sequence_name']).copy()
    for c in ['start','stop','p-value','q-value']:d[c]=pd.to_numeric(d[c],errors='raise')
    if len(d) and (not np.isfinite(d[['start','stop','p-value','q-value']]).all().all() or
        (d.start<1).any() or (d.stop<d.start).any() or not np.isin(d.strand,['+','-']).all() or
        (d[['p-value','q-value']]<0).any().any() or (d[['p-value','q-value']]>1).any().any()):
        raise ValueError('Invalid FIMO coordinates, strand or probabilities')
    return d


def scan_fimo(motif_file, fasta_file, output_dir, *, p_threshold=.001, executable='fimo'):
    """Scan once against the complete prefilter catalog; keep its original q-values."""
    if not 0<p_threshold<=1:raise ValueError('Invalid FIMO p threshold')
    out=Path(output_dir)
    if out.exists():raise FileExistsError(f'FIMO output must be new: {out}')
    command=[str(executable),'--oc',str(out),'--bfile','--uniform--','--thresh',str(p_threshold),
             '--max-stored-scores','3000000',str(motif_file),str(fasta_file)]
    subprocess.run(command,check=True)
    return read_fimo(out/'fimo.tsv')


def distinct_fimo_sites(hits, *, p_threshold=.001, q_threshold=.05):
    """Keep best p among >=70% overlapping same-ID footprints, across strands."""
    if not 0<p_threshold<=1 or not 0<q_threshold<=1:raise ValueError('Invalid scan thresholds')
    d=hits[(hits['p-value']<=p_threshold)&(hits['q-value']<=q_threshold)].copy()
    d['start0']=d.start.astype(int)-1;d['end0']=d.stop.astype(int)
    kept=[]
    for _,g in d.sort_values(['p-value','start','stop','strand']).groupby(['sequence_name','motif_id'],sort=False):
        spans=[]
        for ix,row in g.iterrows():
            a,b=int(row.start0),int(row.end0)
            if any(max(0,min(b,y)-max(a,x))>=.7*min(b-a,y-x) for x,y in spans):continue
            spans.append((a,b));kept.append(ix)
    d=d.loc[kept].copy();d['center0']=(d.start0+d.end0)/2
    return d.reset_index(drop=True)


def select_motif_panel(records, hits, pwms, *, output_dir, top_n=10, min_score=20.,
                       min_sites=30, p_threshold=.001, q_threshold=.05,
                       min_overlap=6, match_similarity=.5,
                       composition_tie=.02, position_tie_bp=2.):
    """Quality gate -> motif composition Max-Min -> Hungarian positions -> score.

    Motifs with fewer than min_sites physical hits across source RECORDS are
    removed. Exact-DNA duplicates contribute according to record multiplicity
    to this support statistic, but are selected only once. No-hit sequences are
    reported and excluded. Unmatched site pairs never create position evidence.
    """
    if isinstance(min_sites,bool) or not isinstance(min_sites,int) or min_sites<1:
        raise ValueError('min_sites must be a positive integer')
    unique,provenance=deduplicate_candidates(records)
    if 'sequence_id' in records and not np.array_equal(records.sequence_id.to_numpy(),provenance.sequence_id.to_numpy()):
        raise ValueError('Sequence IDs must be generated by deduplicate_candidates')
    motif_ids=list(pwms);sequence_ids=unique.sequence_id.tolist()
    if set(hits.motif_id)-set(motif_ids) or set(hits.sequence_name)-set(sequence_ids):
        raise ValueError('FIMO IDs must match the supplied catalog and candidate FASTA')
    length_by_id=unique.set_index('sequence_id').sequence.str.len()
    if len(hits) and (hits.stop.to_numpy()>hits.sequence_name.map(length_by_id).to_numpy()).any():
        raise ValueError('FIMO site is outside its sequence')
    sites=distinct_fimo_sites(hits,p_threshold=p_threshold,q_threshold=q_threshold)
    multiplicity=provenance.sequence_id.value_counts()
    support=sites.assign(source_records=sites.sequence_name.map(multiplicity)).groupby('motif_id').agg(
        sites_on_unique_DNA=('sequence_name','size'),sites_on_source_records=('source_records','sum'),
        unique_DNA_with_hit=('sequence_name','nunique')).reindex(motif_ids,fill_value=0).astype(int)
    support['retained']=support.sites_on_source_records.ge(min_sites)
    motif_ids=[m for m in motif_ids if support.loc[m,'retained']]
    if not motif_ids:raise ValueError('No motifs pass the physical-site support threshold')
    sites=sites[sites.motif_id.isin(motif_ids)].copy()
    mi={m:i for i,m in enumerate(motif_ids)};si={s:i for i,s in enumerate(sequence_ids)}
    presence=np.zeros((len(unique),len(motif_ids)),np.uint8);instances=[[] for _ in sequence_ids]
    for row in sites.itertuples():
        i,j=si[row.sequence_name],mi[row.motif_id]
        presence[i,j]=1;instances[i].append((j,float(row.center0)))
    similarity,alignments=pwm_similarity_matrix([pwms[m] for m in motif_ids],min_overlap=min_overlap)
    composition,fuzzy=soft_jaccard_distance_matrix(presence,similarity)
    position,counts=soft_position_distance_matrix(instances,similarity,min_similarity=match_similarity)
    chosen,trace=select_with_motif_evidence(composition,position,unique.score.to_numpy(),top_n,
        matched_counts=counts,presence=presence,min_specificity=min_score,
        composition_tie=composition_tie,position_tie_bp=position_tie_bp,
        candidate_ids=sequence_ids,require_motif_evidence=True,ignore_unmatched_position_pairs=True)
    if len(chosen)!=top_n:
        raise ValueError(f'Only {len(chosen)} motif-supported sequences pass min_score={min_score}; requested {top_n}')
    selected=unique.iloc[chosen].copy();selected.insert(0,'selection_rank',range(1,len(chosen)+1))
    selected['present_motifs']=[';'.join(np.array(motif_ids)[row.astype(bool)]) for row in presence[chosen]]
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError(f'Use a new empty panel directory: {out}')
    selected.to_csv(out/'selected_sequences.csv',index=False)
    write_fasta(out/'selected_sequences.fa',selected.sequence_id,selected.sequence)
    unique.to_csv(out/'unique_candidates.csv',index=False);provenance.to_csv(out/'source_records.csv',index=False)
    sites.to_csv(out/'motif_sites.csv',index=False);support.to_csv(out/'motif_site_support.csv')
    write_meme(out/'retained_catalog.meme',{m:pwms[m] for m in motif_ids})
    pd.DataFrame(presence,index=pd.Index(sequence_ids,name='sequence_id'),columns=motif_ids).to_csv(out/'presence_matrix.csv')
    expanded=pd.DataFrame(presence[[si[s] for s in provenance.sequence_id]],columns=motif_ids)
    expanded.insert(0,'sequence_id',provenance.sequence_id.to_numpy());expanded.insert(0,'record_id',provenance.record_id.to_numpy())
    expanded.to_csv(out/'all_record_presence_matrix.csv',index=False)
    pd.DataFrame(similarity,index=motif_ids,columns=motif_ids).to_csv(out/'motif_similarity.csv')
    np.savez_compressed(out/'distance_matrices.npz',sequence_ids=np.array(sequence_ids),
        motif_ids=np.array(motif_ids),composition=composition,position_bp=position,
        matched_counts=counts,presence=presence,fuzzy_presence=fuzzy)
    for row in trace:row['sequence_id']=sequence_ids[row['index']]
    (out/'selection_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    receipt=dict(source_records=len(records),unique_DNA=len(unique),selected=top_n,
        no_catalog_hit=int((presence.sum(1)==0).sum()),min_score=min_score,min_sites=min_sites,
        site_support_unit='NMS-filtered physical sites across source records, weighted by exact-DNA multiplicity',
        fimo_p=p_threshold,fimo_q=q_threshold,q_values='unchanged from full prefilter catalog scan',
        composition_tie=composition_tie,position_tie_bp=position_tie_bp,match_similarity=match_similarity,
        positional_missing_evidence='ignore unmatched pairs; no comparable pairs gets no positional advantage')
    (out/'run.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return selected,trace
