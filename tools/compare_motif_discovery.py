"""Reproduce discovery and common family-held-out validation on the shipped seqlets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sequence_optimization.simple_streme import (
    discover_seqlet_motifs, encode_fragments, group_max, optimal_threshold, paired_logp, scan_best,
)
from sequence_optimization.motifs import read_meme, write_meme
from sequence_optimization.motif_diversity import pwm_similarity_matrix
from sequence_optimization.workflows import write_fasta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlets', type=Path, default=ROOT/'examples/data/motif_discovery/seqlets.csv.gz')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=['builtin','streme','pystreme'], default=['builtin','streme'])
    parser.add_argument('--pystreme-source', type=Path)
    parser.add_argument('--streme', default='streme')
    parser.add_argument('--max-motifs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(args.seqlets)
    train = data.split.eq('train').to_numpy()
    groups = data.cdhit90_family.to_numpy()
    if set(groups[train]) & set(groups[~train]):
        raise ValueError('Families cross training and held-out data')
    catalogs = {}
    for method in args.methods:
        folder = args.output_dir/method
        folder.mkdir()
        began = time.perf_counter()
        if method == 'builtin':
            result = discover_seqlet_motifs(data, controls=data.control.tolist(), groups=groups,
                validation_mask=~train, max_motifs=args.max_motifs, seed=args.seed)
            motifs = result.pwms
            result.support.to_csv(folder/'support.csv', index=False)
            result.split.to_csv(folder/'split.csv', index=False)
        elif method == 'streme':
            write_fasta(folder/'primary.fa', data.loc[train,'seqlet_id'], data.loc[train,'dna'])
            write_fasta(folder/'control.fa', data.loc[train,'seqlet_id'], data.loc[train,'control'])
            with (folder/'streme.log').open('w') as log:
                subprocess.run([args.streme, '--dna', '--p', str(folder/'primary.fa'),
                    '--n', str(folder/'control.fa'), '--oc', str(folder/'native'),
                    '--minw','6','--maxw','15','--nmotifs',str(args.max_motifs),
                    '--seed',str(args.seed)], check=True, stdout=log, stderr=subprocess.STDOUT)
            motifs = read_meme(folder/'native/streme.txt')
        else:
            if args.pystreme_source is not None:
                sys.path.insert(0, str(args.pystreme_source/'src'))
            import torch
            from pystreme import MotifDiscovery
            torch.set_num_threads(2)
            length = int(data.dna.str.len().max())
            pad = lambda seqs: [s.ljust(length, 'N') for s in seqs]
            disc = MotifDiscovery.from_sequences(pad(data.loc[train,'dna']), bg_order=0,
                device='cuda' if torch.cuda.is_available() else 'cpu', holdout_frac=0, seed=args.seed)
            found = disc.fit(n_motifs=args.max_motifs, patience=2,
                control=pad(data.loc[train,'control']), widths=range(6,16), n_per_width=3,
                n_iter=10, bg_order=0, holdout_frac=0, n_decoys=0, seed=args.seed)
            motifs = {f'P{i+1:03}': m.pwm.T for i,m in enumerate(found)}
        if not motifs:
            raise ValueError(f'{method} found no candidates')
        write_meme(folder/'candidates.meme', motifs)
        # Common evaluation reads the portable matrices for every backend.
        catalogs[method] = read_meme(folder/'candidates.meme')
        (folder/'runtime.json').write_text(json.dumps(dict(method=method,
            seconds=time.perf_counter()-began, candidates=len(motifs)), indent=2)+'\n')

    positive, negative = encode_fragments(data.dna), encode_fragments(data.control)
    _, labels = np.unique(groups, return_inverse=True)
    train_ids, heldout_ids = np.unique(labels[train]), np.unique(labels[~train])
    train_labels = np.searchsorted(train_ids, labels[train])
    heldout_labels = np.searchsorted(heldout_ids, labels[~train])
    counts = np.bincount(negative[train][negative[train]>=0], minlength=4)+1.
    background = (counts+counts[::-1])/(2*counts.sum())
    hypotheses = sum(map(len, catalogs.values()))
    rows = []
    for method, motifs in catalogs.items():
        for name, pwm in motifs.items():
            a = scan_best(positive, pwm, background)[0]
            b = scan_best(negative, pwm, background)[0]
            threshold, _ = optimal_threshold(group_max(a[train], train_labels, len(train_ids)),
                group_max(b[train], train_labels, len(train_ids)))
            va = group_max(a[~train], heldout_labels, len(heldout_ids))
            vb = group_max(b[~train], heldout_labels, len(heldout_ids))
            p = float(np.exp(paired_logp(va, vb, threshold)))
            support = a >= threshold
            parents = data.loc[support,'sequence_id'].nunique()
            rows.append(dict(method=method, motif_id=name, width=len(pwm),
                consensus=''.join(np.array(list('ACGT'))[pwm.argmax(1)]), threshold_bits=threshold,
                heldout_p=p, common_p_adjusted=min(1., p*hypotheses),
                heldout_positive_groups=int((va>=threshold).sum()),
                heldout_control_groups=int((vb>=threshold).sum()),
                seqlets=int(support.sum()), parents=int(parents),
                retained=bool(p*hypotheses<=.05 and support.sum()>=10 and parents>=3)))
    validation = pd.DataFrame(rows)
    validation.to_csv(args.output_dir/'common_validation.csv', index=False)
    for method, motifs in catalogs.items():
        keep = validation.loc[validation.method.eq(method)&validation.retained, 'motif_id']
        if len(keep):
            write_meme(args.output_dir/method/'validated.meme', {key: motifs[key] for key in keep})
    reference = read_meme(ROOT/'examples/data/motif_discovery/reference.meme')
    matches = []
    for method, motifs in catalogs.items():
        similarity, _ = pwm_similarity_matrix(list(reference.values())+list(motifs.values()))
        for i, key in enumerate(reference):
            j = int(similarity[i,len(reference):].argmax())
            matches.append(dict(method=method, reference=key, best_motif=list(motifs)[j],
                similarity=float(similarity[i,len(reference)+j])))
    pd.DataFrame(matches).to_csv(args.output_dir/'reference_recovery.csv', index=False)
    print(validation.groupby('method').agg(candidates=('motif_id','size'),validated=('retained','sum')))


if __name__ == '__main__':
    main()
