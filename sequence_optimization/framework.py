"""Motif discovery and optional motif-aware candidate selection."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess

import numpy as np
import pandas as pd

from .suite import generate_designs
from .api import SequencePredictor, as_score_batch, normalize_nucleotide_sequences
from .attribution import ism_contributions
from .objectives import MultiTaskObjective, TargetObjective
from .motifs import read_meme, write_meme, scan_fimo, select_motif_panel
from .workflows import deduplicate_candidates, select_sequence_panel, write_fasta


__all__ = ["generate_designs", "call_motifs", "select_candidates", "MotifCatalog"]


@dataclass
class MotifCatalog:
    """PWMs, supporting seqlets, and optional saved sequence hits."""
    pwms: dict
    seqlets: pd.DataFrame = field(default_factory=pd.DataFrame)
    support: pd.DataFrame = field(default_factory=pd.DataFrame)
    hits: pd.DataFrame | None = None
    sequence_ids: tuple = ()
    output_dir: Path | None = None

    def __len__(self):
        return len(self.pwms)


def _records(designs):
    frame = designs.copy() if isinstance(designs, pd.DataFrame) else pd.DataFrame({'sequence': designs})
    frame['sequence'] = normalize_nucleotide_sequences(frame.sequence)
    if frame.empty or frame.sequence.str.len().nunique() != 1:
        raise ValueError('Provide nonempty, equal-length candidate sequences')
    for name, value in [('algorithm', 'input'), ('score', 0.)]:
        if name not in frame:
            frame[name] = value
    if 'record_id' not in frame:
        frame['record_id'] = [f'D{i+1:05}' for i in range(len(frame))]
    unique, frame = deduplicate_candidates(frame)
    return unique, frame


@contextmanager
def _output_folder(output_dir):
    if output_dir is None:
        with TemporaryDirectory(prefix='bioseq_optimization_') as temp:
            yield Path(temp)
    else:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        if any(folder.iterdir()):
            raise FileExistsError(f'Use an empty output directory: {folder}')
        yield folder


def _scorer(model, task_names=None):
    if callable(getattr(model, 'score', None)):
        return model
    import torch
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Supply a PyTorch module or sequence scorer')
    tensor = next(model.parameters(), None)
    if tensor is None:
        tensor = next(model.buffers(), None)
    return SequencePredictor(model, tasks=task_names, device=tensor.device if tensor is not None else 'cpu')


def _objective(designs, objective):
    if objective is None and isinstance(designs, pd.DataFrame):
        objective = designs.attrs.get('objective')
    if objective is None:
        return TargetObjective()
    return MultiTaskObjective(directions=objective) if isinstance(objective, (list, tuple)) else objective


def call_motifs(designs, model=None, objective=None, *, contributions=None,
                catalog=None, hits=None, output_dir=None, batch_size=512,
                min_seqlets=10, min_parents=3, seqlet_threshold=.01,
                min_width=6, max_width=15, max_seqlet_len=25, flanks=2,
                max_motifs=30, seed=42, streme='streme', fimo='fimo'):
    """Discover motifs from optimized sequences and their nucleotide contributions.

    By default, compute ISM for the design objective, call positive tangermeme
    seqlets, discover PWMs with STREME against dinucleotide-shuffled controls,
    and retain models supported by min_seqlets fragments and min_parents DNA.
    Supply contributions as [input records,length] or a DataFrame indexed by
    sequence_id to reuse existing ISM/DeepLIFTShap scores. Alternatively, supply
    a MEME file/PWM dictionary as catalog and optional saved FIMO hits.
    """
    unique, records = _records(designs)
    if min_seqlets < 1 or min_parents < 1 or not 1 <= min_width <= max_width <= max_seqlet_len:
        raise ValueError('Invalid motif support or width parameters')
    ids = tuple(unique.sequence_id)
    saved_dir = None if output_dir is None else Path(output_dir)
    with _output_folder(output_dir) as out:
        write_fasta(out/'sequences.fa', unique.sequence_id, unique.sequence)
        unique.to_csv(out/'sequences.csv', index=False)
        if catalog is not None:
            pwms = read_meme(catalog) if isinstance(catalog, (str, Path)) else dict(catalog)
            write_meme(out/'catalog.meme', pwms)
            if hits is not None:
                if set(hits.sequence_name)-set(ids) or set(hits.motif_id)-set(pwms):
                    raise ValueError('Saved motif hits must match the input sequences and catalog')
                hits.to_csv(out/'saved_hits.csv', index=False)
            return MotifCatalog(pwms, hits=None if hits is None else hits.copy(),
                                sequence_ids=ids, output_dir=saved_dir)

        if contributions is None:
            if model is None:
                raise ValueError('Supply model or precomputed nucleotide contributions')
            scorer = _scorer(model, designs.attrs.get('task_names') if isinstance(designs, pd.DataFrame) else None)
            criterion = _objective(designs, objective)
            def fitness(sequences):
                values = as_score_batch(scorer.score(sequences)).scores
                if values.ndim == 1:
                    values = values[:, None]
                return criterion.scalarize(values, getattr(scorer, 'task_names', None))
            projected = ism_contributions(unique.sequence.to_numpy(), fitness, batch_size=batch_size).observed
        elif isinstance(contributions, pd.DataFrame):
            projected = contributions.loc[list(ids)].to_numpy(dtype=float)
        else:
            values = np.asarray(contributions, dtype=float)
            if values.shape != (len(records), len(unique.sequence.iloc[0])):
                raise ValueError('Contributions must have one row per input record and one column per base')
            first = {sid: i for i, sid in reversed(list(enumerate(records.sequence_id)))}
            projected = values[[first[sid] for sid in ids]]
        expected = (len(unique), len(unique.sequence.iloc[0]))
        if projected.shape != expected or not np.isfinite(projected).all():
            raise ValueError(f'Expected finite projected contributions with shape {expected}')
        np.savez_compressed(out/'contributions.npz', sequence_ids=np.array(ids), observed=projected)

        import torch
        from tangermeme.seqlet import recursive_seqlets
        from tangermeme.ersatz import dinucleotide_shuffle
        from .onehot import onehot
        seqlets = recursive_seqlets(torch.tensor(projected, dtype=torch.float32),
            threshold=seqlet_threshold, min_seqlet_len=min_width,
            max_seqlet_len=max_seqlet_len, additional_flanks=flanks)
        if seqlets.empty:
            raise ValueError('No positive seqlets passed the attribution threshold')
        seqlets['example_idx'] = seqlets.example_idx.astype(int)
        seqlets['start'] = seqlets.start.astype(int).clip(lower=0)
        seqlets['end'] = seqlets.end.astype(int).clip(upper=expected[1])
        seqlets['sequence_id'] = seqlets.example_idx.map(unique.sequence_id.to_dict())
        seqlets['dna'] = [unique.sequence.iloc[r.example_idx][r.start:r.end] for r in seqlets.itertuples()]
        seqlets = seqlets[seqlets.dna.str.len().ge(min_width) & seqlets.attribution.gt(0)].copy()
        seqlets['seqlet_id'] = [f'Q{i+1:06}' for i in range(len(seqlets))]
        seqlets.to_csv(out/'seqlets.csv', index=False)
        if len(seqlets) < min_seqlets:
            raise ValueError(f'Found {len(seqlets)} positive seqlets; require at least {min_seqlets}')
        write_fasta(out/'seqlets.fa', seqlets.seqlet_id, seqlets.dna)
        controls = {}
        for _, group in seqlets.groupby(seqlets.dna.str.len()):
            tensor = torch.tensor(onehot(group.dna.to_numpy(), channels_first=True))
            shuffled = dinucleotide_shuffle(tensor, n=1, random_state=seed)
            codes = shuffled[:, 0].argmax(dim=1).cpu().numpy()
            controls.update({sid: ''.join(np.array(list('ACGT'))[row]) for sid, row in zip(group.seqlet_id, codes)})
        write_fasta(out/'background.fa', seqlets.seqlet_id, [controls[sid] for sid in seqlets.seqlet_id])
        command = [str(streme), '--dna', '--p', str(out/'seqlets.fa'), '--n', str(out/'background.fa'),
                   '--oc', str(out/'streme'), '--minw', str(min_width), '--maxw', str(max_width),
                   '--nmotifs', str(max_motifs), '--seed', str(seed)]
        with (out/'streme.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        discovered = read_meme(out/'streme/streme.txt')
        seqlet_hits = scan_fimo(out/'streme/streme.txt', out/'seqlets.fa', out/'seqlet_scan', executable=fimo)
        support = []
        for motif_id in discovered:
            matched = set(seqlet_hits.loc[seqlet_hits.motif_id.eq(motif_id), 'sequence_name'])
            group = seqlets[seqlets.seqlet_id.isin(matched)]
            support.append(dict(motif_id=motif_id, seqlets=len(group), parents=group.sequence_id.nunique()))
        support = pd.DataFrame(support)
        support['retained'] = support.seqlets.ge(min_seqlets) & support.parents.ge(min_parents)
        support.to_csv(out/'support.csv', index=False)
        pwms = {m: discovered[m] for m in support.loc[support.retained, 'motif_id']}
        if not pwms:
            raise ValueError('No motifs pass the seqlet and parent support thresholds')
        write_meme(out/'catalog.meme', pwms)
        (out/'config.json').write_text(json.dumps(dict(method='ISM' if contributions is None else 'provided',
            sequences=len(unique), seqlets=len(seqlets), motifs=len(pwms), min_seqlets=min_seqlets,
            min_parents=min_parents, seqlet_threshold=seqlet_threshold, seed=seed), indent=2)+'\n')
        return MotifCatalog(pwms, seqlets=seqlets, support=support, sequence_ids=ids, output_dir=saved_dir)


def select_candidates(designs, top_n=10, motifs=None, *, min_score=None,
                      output_dir=None, distance_tie_nt=1, min_sites=30,
                      match_similarity=.5, composition_tie=.02,
                      position_tie_bp=2., fimo='fimo'):
    """Select a high-quality diverse panel, returning one DataFrame in either mode.

    motifs=None uses Hamming Max-Min. Supplying a MotifCatalog, MEME file, or PWM
    dictionary uses motif-composition Max-Min, Hungarian positional Max-Min,
    then prediction score. The optional output directory receives the panel,
    FASTA, distances and selection trace.
    """
    unique, records = _records(designs)
    gate = float(unique.score.min()) if min_score is None else min_score
    if motifs is None:
        selected, distance = select_sequence_panel(records, top_n=top_n,
            min_score=gate, distance_tie_nt=distance_tie_nt)
        trace = [dict(rank=i+1, sequence_id=sid, score=float(selected.score.iloc[i]),
                      minimum_hamming=None if i == 0 else int(distance[i, :i].min()))
                 for i, sid in enumerate(selected.sequence_id)]
        selected.attrs.update(selection_method='hamming', selection_trace=trace,
                              hamming_distances=distance.tolist())
        if output_dir is not None:
            with _output_folder(output_dir) as out:
                selected.to_csv(out/'selected_sequences.csv', index=False)
                write_fasta(out/'selected_sequences.fa', selected.sequence_id, selected.sequence)
                np.savetxt(out/'hamming_distances.csv', distance, delimiter=',', fmt='%d')
                (out/'selection_trace.json').write_text(json.dumps(trace, indent=2)+'\n')
                (out/'config.json').write_text(json.dumps(dict(method='hamming', top_n=top_n,
                    min_score=gate, distance_tie_nt=distance_tie_nt), indent=2)+'\n')
        return selected

    if isinstance(motifs, MotifCatalog):
        pwms = motifs.pwms
        hits = motifs.hits if set(motifs.sequence_ids) == set(unique.sequence_id) else None
    else:
        pwms = read_meme(motifs) if isinstance(motifs, (str, Path)) else dict(motifs)
        hits = None
    if not pwms:
        raise ValueError('Supply a nonempty motif catalog')
    with _output_folder(output_dir) as out:
        write_meme(out/'input_catalog.meme', pwms)
        write_fasta(out/'input_sequences.fa', unique.sequence_id, unique.sequence)
        if hits is None:
            hits = scan_fimo(out/'input_catalog.meme', out/'input_sequences.fa', out/'fimo', executable=fimo)
        panel, trace = select_motif_panel(records, hits, pwms, output_dir=out/'panel',
            top_n=top_n, min_score=gate, min_sites=min_sites,
            match_similarity=match_similarity, composition_tie=composition_tie,
            position_tie_bp=position_tie_bp)
        retained_hits = pd.read_csv(out/'panel/motif_sites.csv')
        panel.attrs.update(selection_method='motif', selection_trace=trace,
            motif_instances=retained_hits[retained_hits.sequence_name.isin(panel.sequence_id)].to_dict('records'))
        # Keep the public output layout identical between the two selection modes.
        if output_dir is not None:
            import shutil
            for path in (out/'panel').iterdir():
                shutil.move(str(path), out/path.name)
            (out/'panel').rmdir()
        return panel
