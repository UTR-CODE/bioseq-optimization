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

class MemeMotifs(dict):
    """PWM dictionary retaining native MEME blocks for reproducible FIMO scans."""

    def __init__(self, *args, preamble=None, blocks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.preamble = preamble
        self.blocks = blocks or {}


def subset_motifs(motifs, ids):
    ids = list(ids)
    return MemeMotifs({m: motifs[m] for m in ids},
                      preamble=getattr(motifs, 'preamble', None),
                      blocks={m: motifs.blocks[m] for m in ids if m in getattr(motifs, 'blocks', {})})


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
    starts = [i for i, line in enumerate(lines) if line.startswith('MOTIF ')]
    blocks = {lines[i].split()[1]: '\n'.join(lines[i:starts[j+1] if j+1 < len(starts) else len(lines)])
              for j, i in enumerate(starts)}
    return MemeMotifs(motifs, preamble='\n'.join(lines[:starts[0]]), blocks=blocks)


def write_meme(path, motifs):
    blocks = getattr(motifs, 'blocks', {})
    def unchanged(m):
        if m not in blocks:
            return False
        rows = blocks[m].splitlines()
        i = next(j for j, line in enumerate(rows) if line.startswith('letter-probability matrix:'))
        width = int(re.search(r'\bw\s*=\s*(\d+)', rows[i]).group(1))
        native = np.array([[float(v) for v in row.split()] for row in rows[i+1:i+1+width]])
        native /= native.sum(1, keepdims=True)
        return np.array_equal(np.asarray(motifs[m]), native)

    if getattr(motifs, 'preamble', None) is not None and all(unchanged(m) for m in motifs):
        # Reuse native matrices, nsites, background and discovery statistics.
        Path(path).write_text(motifs.preamble + '\n' + '\n'.join(blocks[m] for m in motifs) + '\n')
        return
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


def scan_fimo(motif_file, fasta_file, output_dir, *, p_threshold=.001, executable='fimo',
              reverse_complement=True):
    """Scan once against the complete prefilter catalog; keep its original q-values."""
    if not 0<p_threshold<=1:raise ValueError('Invalid FIMO p threshold')
    if not isinstance(reverse_complement, (bool, np.bool_)):
        raise ValueError('reverse_complement must be a boolean')
    out=Path(output_dir)
    if out.exists():raise FileExistsError(f'FIMO output must be new: {out}')
    command=[str(executable),'--oc',str(out),'--bfile','--uniform--','--thresh',str(p_threshold),
             '--max-stored-scores','3000000',str(motif_file),str(fasta_file)]
    if not reverse_complement:command.insert(1,'--norc')
    subprocess.run(command,check=True)
    return read_fimo(out/'fimo.tsv')


def distinct_fimo_sites(hits, *, p_threshold=.001, q_threshold=.1):
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


def filter_motif_sites(sites, pwms, *, min_sites=30, subset_threshold=.8):
    """Require >min_sites unique-DNA sites; remove spatially redundant motifs.

    Positions match on the same DNA when overlap covers >=80% of the shorter
    footprint, irrespective of strand. A motif is redundant when either
    directional covered-site fraction exceeds subset_threshold. Greedy
    representatives prioritize site count, PWM information, then stable ID.
    """
    if isinstance(min_sites, bool) or not isinstance(min_sites, int) or min_sites < 0:
        raise ValueError('min_sites must be a nonnegative integer')
    if not 0 < subset_threshold <= 1:
        raise ValueError('subset_threshold must be in (0, 1]')
    counts = sites.groupby('motif_id').size().reindex(list(pwms), fill_value=0)
    support = pd.DataFrame({'sites_on_unique_DNA': counts})
    support['passes_site_support'] = counts.gt(min_sites)
    support['retained'] = False
    support['redundant_with'] = ''
    spans = {m: {sid: g[['start0', 'end0']].to_numpy()
                 for sid, g in group.groupby('sequence_name')}
             for m, group in sites.groupby('motif_id')}

    def information(m):
        p = np.asarray(pwms[m], dtype=float)
        return float((2 + (p * np.log2(np.maximum(p, 1e-300))).sum(1)).sum())

    ranked = sorted(counts[counts.gt(min_sites)].index,
                    key=lambda m: (-int(counts[m]), -information(m), m))
    kept, comparisons = [], []
    for m in ranked:
        for representative in kept:
            covered_m = covered_r = 0
            for sid in spans[m].keys() & spans[representative].keys():
                a, b = spans[m][sid], spans[representative][sid]
                overlap = np.maximum(0, np.minimum(a[:, None, 1], b[None, :, 1])
                                     - np.maximum(a[:, None, 0], b[None, :, 0]))
                matched = overlap >= .8 * np.minimum(
                    (a[:, 1]-a[:, 0])[:, None], (b[:, 1]-b[:, 0])[None, :])
                covered_m += int(matched.any(1).sum())
                covered_r += int(matched.any(0).sum())
            fraction_m = covered_m / counts[m]
            fraction_r = covered_r / counts[representative]
            redundant = max(fraction_m, fraction_r) > subset_threshold
            comparisons.append(dict(motif_id=m, representative=representative,
                motif_covered_fraction=fraction_m, representative_covered_fraction=fraction_r,
                redundant=redundant))
            if redundant:
                support.loc[m, 'redundant_with'] = representative
                break
        else:
            kept.append(m)
            support.loc[m, 'retained'] = True
    columns = ['motif_id', 'representative', 'motif_covered_fraction',
               'representative_covered_fraction', 'redundant']
    return (sites[sites.motif_id.isin(kept)].copy(), support,
            pd.DataFrame(comparisons, columns=columns))


def select_motif_panel(records, hits, pwms, *, output_dir, top_n=10, min_score=20.,
                       min_sites=30, p_threshold=.001, q_threshold=.1,
                       min_overlap=6, match_similarity=.5,
                       composition_tie=.02, position_tie_bp=2., subset_threshold=.8,
                       reverse_complement=True):
    """Quality gate -> motif composition Max-Min -> Hungarian positions -> score.

    Motifs must have >min_sites physical hits on unique DNA. Motifs with
    redundant site sets are removed before distances are calculated. No-hit sequences are
    reported and excluded. Unmatched site pairs never create position evidence.
    """
    if isinstance(min_sites,bool) or not isinstance(min_sites,int) or min_sites<0:
        raise ValueError('min_sites must be a nonnegative integer')
    if not isinstance(reverse_complement, (bool, np.bool_)):
        raise ValueError('reverse_complement must be a boolean')
    if not reverse_complement and not hits.strand.eq('+').all():
        raise ValueError('Forward-only selection requires hits from a forward-only scan; rescan instead of filtering two-strand q-values')
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
    extra_support=sites.assign(source_records=sites.sequence_name.map(multiplicity)).groupby('motif_id').agg(
        sites_on_source_records=('source_records','sum'),
        unique_DNA_with_hit=('sequence_name','nunique')).reindex(motif_ids,fill_value=0).astype(int)
    sites,support,redundancy=filter_motif_sites(sites,pwms,min_sites=min_sites,
                                              subset_threshold=subset_threshold)
    support=support.join(extra_support)
    motif_ids=[m for m in motif_ids if support.loc[m,'retained']]
    if not motif_ids:raise ValueError('No motifs pass the physical-site support threshold')
    sites=sites[sites.motif_id.isin(motif_ids)].copy()
    mi={m:i for i,m in enumerate(motif_ids)};si={s:i for i,s in enumerate(sequence_ids)}
    presence=np.zeros((len(unique),len(motif_ids)),np.uint8);instances=[[] for _ in sequence_ids]
    for row in sites.itertuples():
        i,j=si[row.sequence_name],mi[row.motif_id]
        presence[i,j]=1;instances[i].append((j,float(row.center0)))
    similarity,alignments=pwm_similarity_matrix([pwms[m] for m in motif_ids],min_overlap=min_overlap,
                                               reverse_complement=reverse_complement)
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
    redundancy.to_csv(out/'motif_redundancy.csv',index=False)
    write_meme(out/'retained_catalog.meme',subset_motifs(pwms, motif_ids))
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
        reverse_complement=bool(reverse_complement),
        no_catalog_hit=int((presence.sum(1)==0).sum()),min_score=min_score,min_sites=min_sites,
        site_support_unit='NMS-filtered physical sites on unique DNA; strictly greater than min_sites',
        subset_threshold=subset_threshold,site_overlap_fraction=.8,
        fimo_p=p_threshold,fimo_q=q_threshold,q_values='unchanged from full prefilter catalog scan',
        composition_tie=composition_tie,position_tie_bp=position_tie_bp,match_similarity=match_similarity,
        positional_missing_evidence='ignore unmatched pairs; no comparable pairs gets no positional advantage')
    (out/'run.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return selected,trace
