"""Two-pass motif discovery used by the Malinois Figure 2 example."""

import json
import re
import subprocess
import shutil
from pathlib import Path

import pandas as pd
from MotifSift import load_motifs
from sequence_optimization.motifs import (
    MemeMotifs, distinct_fimo_sites, filter_motif_sites, read_meme,
    scan_fimo, subset_motifs, write_meme,
)
from sequence_optimization.simple_streme import dinucleotide_controls
from sequence_optimization.workflows import write_fasta

def combine_seqlet_motifs(first_pass, output):
    """Run a second seed-42 discovery pass and filter both passes on the full pool."""
    first_pass, output = Path(first_pass), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reverse_complement=str.maketrans('ACGT','TGCA')
    seqlets=pd.read_csv(first_pass/'seqlets.csv')
    balanced=seqlets.sort_values(['sequence_id','start','end'],kind='stable').copy()
    balanced['canonical']=balanced.dna.map(lambda s:min(s,s.translate(reverse_complement)[::-1]))
    balanced=balanced.drop_duplicates('canonical')
    write_fasta(output/'balanced_seqlets.fa',balanced.seqlet_id,balanced.dna)
    write_fasta(output/'balanced_background.fa',balanced.seqlet_id,dinucleotide_controls(balanced.dna.tolist(),seed=42))

    with (output/'xstreme.log').open('w') as log:
        subprocess.run(['xstreme','--dna','--p',str(output/'balanced_seqlets.fa'),
            '--n',str(output/'balanced_background.fa'),'--order','1','--oc',str(output/'xstreme'),
            '--minw','6','--maxw','15','--streme-nmotifs','30','--meme-nmotifs','0','--seed','42','--fimo-skip'],
            check=True,stdout=log,stderr=subprocess.STDOUT)

    baseline=read_meme(first_pass/'xstreme/xstreme.txt')
    additional=read_meme(output/'xstreme/xstreme.txt')
    pool=MemeMotifs(dict(baseline),preamble=baseline.preamble,blocks=dict(baseline.blocks))
    for motif,pwm in additional.items():
        name=f'balanced-{motif}'
        pool[name]=pwm
        pool.blocks[name]=re.sub(r'^MOTIF \S+',f'MOTIF {name}',additional.blocks[motif],count=1)
    write_meme(output/'combined.meme',pool)
    seqlet_hits=scan_fimo(output/'combined.meme', first_pass/'seqlets.fa', output/'seqlet_scan',executable='fimo')
    counts=[]
    for motif in pool:
        matched=set(seqlet_hits.loc[seqlet_hits.motif_id.eq(motif),'sequence_name'])
        group=seqlets[seqlets.seqlet_id.isin(matched)]
        counts.append(dict(motif_id=motif,seqlets=len(group),parents=group.sequence_id.nunique()))
    support=pd.DataFrame(counts)
    support['passes_seqlet_support']=support.seqlets.ge(10)&support.parents.ge(3)
    eligible=subset_motifs(pool,support.loc[support.passes_seqlet_support,'motif_id'])
    candidate_hits=scan_fimo(output/'combined.meme', first_pass/'sequences.fa', output/'candidate_scan',executable='fimo')
    candidate_hits=candidate_hits[candidate_hits.motif_id.isin(eligible)]
    sites, by_motif, duplicate=filter_motif_sites(distinct_fimo_sites(candidate_hits),eligible)
    support.join(by_motif,on='motif_id').to_csv(output/'support.csv',index=False)
    sites.to_csv(output/'motif_sites.csv',index=False)
    duplicate.to_csv(output/'motif_redundancy.csv',index=False)
    final=subset_motifs(eligible,by_motif.index[by_motif.retained])
    write_meme(output/'catalog.meme',final)
    report=dict(original_seqlets=len(seqlets),unique_seqlets_in_second_pass=len(balanced),
                original_raw_motifs=len(baseline),second_pass_raw_motifs=len(additional),
                combined_final_motifs=len(final),final_ids=list(final))
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in ("seqlets.csv", "seqlets.fa", "sequences.csv", "sequences.fa", "contributions.npz"):
        shutil.copyfile(first_pass / name, output / name)
    candidate_hits[candidate_hits.motif_id.isin(final)].to_csv(output / "saved_hits.csv", index=False)
    config = json.loads((first_pass / "config.json").read_text())
    config.update(motifs=len(final), discovery_passes=["all seqlets", "unique reverse-complement seqlets"])
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return load_motifs(output)
