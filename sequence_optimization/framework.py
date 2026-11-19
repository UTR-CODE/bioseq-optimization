"""Motif discovery and optional motif-aware candidate selection."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

from .suite import generate_designs
from .api import SequencePredictor, as_score_batch, normalize_nucleotide_sequences
from .attribution import deepliftshap_contributions, ism_contributions, synonymous_ism_contributions
from .objectives import MultiTaskObjective, TargetObjective
from .motifs import (read_meme, write_meme, scan_fimo, select_motif_panel,
                     distinct_fimo_sites, filter_motif_sites, subset_motifs)
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
    p_threshold: float = .001
    q_threshold: float = .1
    reverse_complement: bool = True

    def __post_init__(self):
        if not isinstance(self.reverse_complement, (bool, np.bool_)):
            raise ValueError('reverse_complement must be a boolean')
        self.reverse_complement = bool(self.reverse_complement)
        if not self.reverse_complement and self.hits is not None and not self.hits.strand.eq('+').all():
            raise ValueError('Forward-only catalogs require forward-only FIMO hits; rescan with reverse_complement=False')

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


def _require_meme_tools(discovery, fimo):
    missing = [f"{name} ({command})" for name, command in
               [('motif discovery', discovery), ('FIMO', fimo)]
               if command is not None and shutil.which(str(command)) is None]
    if missing:
        raise FileNotFoundError(
            "MEME Suite tools are missing or not executable: " + ', '.join(missing)
            + "\nInstall MEME Suite in your active environment:\n"
            "  conda install -c conda-forge -c bioconda meme\n"
            "Then ensure xstreme and fimo are on PATH, or pass their executable "
            "paths using call_motifs(..., xstreme='/path/to/xstreme', fimo='/path/to/fimo')."
        )


def _dna_encoded_rna_catalog(source, destination):
    """Encode RNA as DNA for FIMO without altering native motif statistics."""
    lines = Path(source).read_text().splitlines()
    if [line.replace(' ', '') for line in lines if line.startswith('ALPHABET=')] != ['ALPHABET=ACGU']:
        raise ValueError('Forward-only external discovery must produce an A/C/G/U catalog')
    converted = []
    in_background = False
    for line in lines:
        if line.startswith('ALPHABET='):
            converted.extend(['ALPHABET= ACGT', '', 'strands: +'])
        elif line.startswith('strands:'):
            continue
        elif line.startswith('Background letter frequencies'):
            in_background = True
            converted.append(line)
        elif in_background and re.fullmatch(r'\s*(?:[ACGU]\s+[0-9.eE+\-]+\s*)+', line):
            converted.append(re.sub(r'\bU(?=\s)', 'T', line))
        else:
            if line.strip():
                in_background = False
            converted.append(line)
    Path(destination).write_text('\n'.join(converted)+'\n')


def call_motifs(designs, model=None, objective=None, *, contributions=None,
                catalog=None, hits=None, output_dir=None, batch_size=512,
                min_seqlets=10, min_parents=3, seqlet_threshold=.01,
                min_width=6, max_width=15, max_seqlet_len=25, flanks=2,
                max_motifs=30, seed=42, streme='streme', fimo='fimo',
                motif_backend='xstreme', discovery_options=None, xstreme='xstreme',
                min_sites=30, subset_threshold=.8, p_threshold=.001, q_threshold=.1,
                attribution_method='ism', attribution_model=None,
                n_references=20, attribution_batch_size=64,
                reverse_complement=True):
    """Discover motifs from optimized sequences and their nucleotide contributions.

    By default, compute ISM for the design objective, call positive tangermeme
    seqlets, discover PWMs with official XSTREME against dinucleotide-shuffled
    controls (MEME disabled), and use FIMO to retain models supported by min_seqlets fragments
    and min_parents DNA. Candidate-wide FIMO scanning then requires >min_sites
    physical sites and removes motifs with redundant site sets (>80% by default).
    Set attribution_method="deepliftshap" for differentiable attribution using
    n_references dinucleotide-shuffled references. attribution_model optionally
    supplies a scalar-objective [N,4,L] -> [N,1] module for custom models.
    Candidate FIMO defaults to p<=0.001 and q<=0.1.
    motif_backend="builtin" selects the optional Python
    implementation with independent-group held-out enrichment.
    Supply contributions as [input records,length] or a DataFrame indexed by
    sequence_id to reuse existing ISM/DeepLIFTShap scores. Alternatively, supply
    a MEME file/PWM dictionary as catalog and optional saved FIMO hits.
    Set reverse_complement=False for RNA: discover and scan only the supplied
    orientation. The returned catalog carries this setting into selection.
    This setting does not change strand handling inside a user-supplied scorer.
    """
    if not isinstance(reverse_complement, (bool, np.bool_)):
        raise ValueError('reverse_complement must be a boolean')
    reverse_complement = bool(reverse_complement)
    if 'reverse_complement' in (discovery_options or {}):
        raise ValueError('Pass reverse_complement directly to call_motifs, not discovery_options')
    attribution_method = str(attribution_method).lower().replace('_', '')
    if attribution_method not in {'ism', 'deepliftshap'}:
        raise ValueError("attribution_method must be 'ism' or 'deepliftshap'")
    if not 0 < p_threshold <= 1 or not 0 < q_threshold <= 1:
        raise ValueError('FIMO thresholds must be in (0, 1]')
    unique, records = _records(designs)
    if min_seqlets < 1 or min_parents < 1 or not 1 <= min_width <= max_width <= max_seqlet_len:
        raise ValueError('Invalid motif support or width parameters')
    if motif_backend not in {'builtin', 'streme', 'xstreme'}:
        raise ValueError("motif_backend must be 'builtin', 'streme' or 'xstreme'")
    if isinstance(min_sites, bool) or not isinstance(min_sites, int) or min_sites < 0:
        raise ValueError('min_sites must be a nonnegative integer')
    if not 0 < subset_threshold <= 1:
        raise ValueError('subset_threshold must be in (0, 1]')
    if catalog is None:
        discovery_tool = xstreme if motif_backend == 'xstreme' else streme if motif_backend == 'streme' else None
        _require_meme_tools(discovery_tool, fimo)
    ids = tuple(unique.sequence_id)
    fixed_cds = designs.attrs.get('cds_fixed_positions') if isinstance(designs, pd.DataFrame) else None
    method = attribution_method if contributions is None else 'provided'
    saved_dir = None if output_dir is None else Path(output_dir)
    with _output_folder(output_dir) as out:
        write_fasta(out/'sequences.fa', unique.sequence_id, unique.sequence)
        unique.to_csv(out/'sequences.csv', index=False)
        if catalog is not None:
            pwms = read_meme(catalog) if isinstance(catalog, (str, Path)) else subset_motifs(catalog, catalog)
            write_meme(out/'catalog.meme', pwms)
            if hits is not None:
                if set(hits.sequence_name)-set(ids) or set(hits.motif_id)-set(pwms):
                    raise ValueError('Saved motif hits must match the input sequences and catalog')
                if not reverse_complement and not hits.strand.eq('+').all():
                    raise ValueError('Forward-only mode requires forward-only FIMO hits; omit hits to rescan')
                hits.to_csv(out/'saved_hits.csv', index=False)
            (out/'config.json').write_text(json.dumps(dict(method='provided_catalog',
                reverse_complement=reverse_complement, fimo_p=p_threshold, fimo_q=q_threshold), indent=2)+'\n')
            return MotifCatalog(pwms, hits=None if hits is None else hits.copy(),
                                sequence_ids=ids, output_dir=saved_dir,
                                p_threshold=p_threshold, q_threshold=q_threshold,
                                reverse_complement=reverse_complement)

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
            if attribution_method == 'ism' and fixed_cds is not None:
                method = 'synonymous_ism'
                projected = synonymous_ism_contributions(
                    unique.sequence.to_numpy(), fitness, fixed_positions=fixed_cds, batch_size=batch_size,
                )
            elif attribution_method == 'ism':
                projected = ism_contributions(unique.sequence.to_numpy(), fitness, batch_size=batch_size).observed
            else:
                projected, diagnostics = deepliftshap_contributions(
                    unique.sequence.to_numpy(), scorer, criterion,
                    attribution_model=attribution_model, n_references=n_references,
                    batch_size=attribution_batch_size, seed=seed)
                (out/'attribution.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
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
        from .simple_streme import dinucleotide_controls, discover_seqlet_motifs
        controls = dinucleotide_controls(seqlets.dna.tolist(), seed)
        write_fasta(out/'background.fa', seqlets.seqlet_id, controls)
        if motif_backend == 'builtin':
            discovery = discover_seqlet_motifs(seqlets, controls=controls,
                min_width=min_width, max_width=max_width, max_motifs=max_motifs,
                min_seqlets=min_seqlets, min_parents=min_parents, seed=seed,
                reverse_complement=reverse_complement,
                **dict(discovery_options or {}))
            discovered, support = discovery.pwms, discovery.support
            discovery.split.to_csv(out/'discovery_split.csv', index=False)
            if discovered:
                write_meme(out/'candidate_motifs.meme', discovered)
        else:
            if motif_backend == 'xstreme':
                command = [str(xstreme), '--dna', '--p', str(out/'seqlets.fa'),
                           '--n', str(out/'background.fa'), '--order', '1',
                           '--oc', str(out/'xstreme'), '--minw', str(min_width),
                           '--maxw', str(max_width), '--streme-nmotifs', str(max_motifs),
                           '--meme-nmotifs', '0', '--seed', str(seed), '--fimo-skip']
                motif_file = out/'xstreme/xstreme.txt'
                if not reverse_complement:
                    command.append('--dna2rna')
            else:
                command = [str(streme), '--dna', '--p', str(out/'seqlets.fa'), '--n', str(out/'background.fa'),
                           '--oc', str(out/'streme'), '--minw', str(min_width), '--maxw', str(max_width),
                           '--nmotifs', str(max_motifs), '--seed', str(seed)]
                motif_file = out/'streme/streme.txt'
                if not reverse_complement:
                    command[1] = '--rna'
                    for option, sequences in [('--p', seqlets.dna), ('--n', controls)]:
                        target = out/('seqlets_rna.fa' if option == '--p' else 'background_rna.fa')
                        write_fasta(target, seqlets.seqlet_id, [s.replace('T', 'U') for s in sequences])
                        command[command.index(option)+1] = str(target)
            with (out/f'{motif_backend}.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            if not reverse_complement:
                native_motif_file = motif_file
                motif_file = motif_file.with_name(motif_file.stem+'_dna_encoded.meme')
                _dna_encoded_rna_catalog(native_motif_file, motif_file)
            discovered = read_meme(motif_file)
            seqlet_hits = scan_fimo(motif_file, out/'seqlets.fa', out/'seqlet_scan', executable=fimo,
                                   p_threshold=p_threshold, reverse_complement=reverse_complement)
            rows = []
            for motif_id in discovered:
                matched = set(seqlet_hits.loc[seqlet_hits.motif_id.eq(motif_id), 'sequence_name'])
                group = seqlets[seqlets.seqlet_id.isin(matched)]
                rows.append(dict(motif_id=motif_id, seqlets=len(group), parents=group.sequence_id.nunique()))
            support = pd.DataFrame(rows)
            support['retained'] = support.seqlets.ge(min_seqlets) & support.parents.ge(min_parents)
        support.to_csv(out/'support.csv', index=False)
        pwms = subset_motifs(discovered, support.loc[support.retained, 'motif_id'])
        if not pwms:
            raise ValueError('No motifs pass the discovery significance and seqlet/parent support thresholds; see support.csv')
        write_meme(out/'supported_catalog.meme', pwms)
        # Scan the full native discovery catalog before applying support filters.
        scan_catalog = out/'candidate_motifs.meme' if motif_backend == 'builtin' else motif_file
        candidate_hits = scan_fimo(scan_catalog, out/'sequences.fa',
                                   out/'candidate_scan', executable=fimo, p_threshold=p_threshold,
                                   reverse_complement=reverse_complement)
        candidate_hits = candidate_hits[candidate_hits.motif_id.isin(pwms)].copy()
        sites, site_support, redundancy = filter_motif_sites(
            distinct_fimo_sites(candidate_hits, p_threshold=p_threshold, q_threshold=q_threshold), pwms,
            min_sites=min_sites, subset_threshold=subset_threshold)
        site_support.to_csv(out/'motif_site_support.csv')
        redundancy.to_csv(out/'motif_redundancy.csv', index=False)
        sites.to_csv(out/'motif_sites.csv', index=False)
        pwms = subset_motifs(pwms, [m for m in pwms if site_support.loc[m, 'retained']])
        if not pwms:
            raise ValueError('No motifs pass candidate-site filtering; see motif_site_support.csv')
        support = support.rename(columns={'retained': 'passes_seqlet_support'})
        support = support.join(site_support, on='motif_id')
        support['retained'] = support.retained.fillna(False).astype(bool)
        support.to_csv(out/'support.csv', index=False)
        write_meme(out/'catalog.meme', pwms)
        (out/'config.json').write_text(json.dumps(dict(method=method,
            motif_backend=motif_backend, discovery_options=dict(discovery_options or {}),
            reverse_complement=reverse_complement,
            discovery_alphabet='DNA' if reverse_complement or motif_backend == 'builtin' else 'RNA',
            catalog_encoding='ACGT (U represented as T for RNA)',
            sequences=len(unique), seqlets=len(seqlets), motifs=len(pwms), min_seqlets=min_seqlets,
            min_parents=min_parents, seqlet_threshold=seqlet_threshold, seed=seed,
            min_sites=min_sites, subset_threshold=subset_threshold, site_overlap_fraction=.8,
            site_support_unit='physical sites on unique DNA; strictly greater than min_sites',
            fimo_p=p_threshold, fimo_q=q_threshold, meme_enabled=False,
            n_references=n_references if attribution_method == 'deepliftshap' else None,
            attribution_batch_size=attribution_batch_size,
            native_meme_metadata_preserved=True), indent=2, default=str)+'\n')
        return MotifCatalog(pwms, seqlets=seqlets, support=support,
            hits=candidate_hits[candidate_hits.motif_id.isin(pwms)].copy(),
            sequence_ids=ids, output_dir=saved_dir,
            p_threshold=p_threshold, q_threshold=q_threshold,
            reverse_complement=reverse_complement)


def select_candidates(designs, top_n=10, motifs=None, *, min_score=None,
                      output_dir=None, distance_tie_nt=1, min_sites=30,
                      match_similarity=.5, composition_tie=.02,
                      position_tie_bp=2., fimo='fimo', subset_threshold=.8,
                      p_threshold=None, q_threshold=None, reverse_complement=None):
    """Select a high-quality diverse panel, returning one DataFrame in either mode.

    motifs=None uses Hamming Max-Min. Supplying a MotifCatalog, MEME file, or PWM
    dictionary uses motif-composition Max-Min, Hungarian positional Max-Min,
    then prediction score. The optional output directory receives the panel,
    FASTA, distances and selection trace.
    reverse_complement=None inherits the MotifCatalog setting, otherwise defaults
    to True. Set False with a MEME path/PWM dictionary for forward-only RNA.
    Changing orientation relative to saved catalog hits triggers a fresh scan.
    """
    if reverse_complement is not None and not isinstance(reverse_complement, (bool, np.bool_)):
        raise ValueError('reverse_complement must be a boolean or None')
    if reverse_complement is None:
        reverse_complement = motifs.reverse_complement if isinstance(motifs, MotifCatalog) else True
    reverse_complement = bool(reverse_complement)
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

    p_threshold = (motifs.p_threshold if isinstance(motifs, MotifCatalog) else .001) if p_threshold is None else p_threshold
    q_threshold = (motifs.q_threshold if isinstance(motifs, MotifCatalog) else .1) if q_threshold is None else q_threshold
    if isinstance(motifs, MotifCatalog):
        pwms = motifs.pwms
        hits = motifs.hits if (set(motifs.sequence_ids) == set(unique.sequence_id)
                              and p_threshold <= motifs.p_threshold
                              and reverse_complement == motifs.reverse_complement) else None
    else:
        pwms = read_meme(motifs) if isinstance(motifs, (str, Path)) else subset_motifs(motifs, motifs)
        hits = None
    if not pwms:
        raise ValueError('Supply a nonempty motif catalog')
    with _output_folder(output_dir) as out:
        write_meme(out/'input_catalog.meme', pwms)
        write_fasta(out/'input_sequences.fa', unique.sequence_id, unique.sequence)
        if hits is None:
            hits = scan_fimo(out/'input_catalog.meme', out/'input_sequences.fa', out/'fimo', executable=fimo,
                             p_threshold=p_threshold, reverse_complement=reverse_complement)
        panel, trace = select_motif_panel(records, hits, pwms, output_dir=out/'panel',
            top_n=top_n, min_score=gate, min_sites=min_sites,
            match_similarity=match_similarity, composition_tie=composition_tie,
            position_tie_bp=position_tie_bp, subset_threshold=subset_threshold,
            p_threshold=p_threshold, q_threshold=q_threshold,
            reverse_complement=reverse_complement)
        retained_hits = pd.read_csv(out/'panel/motif_sites.csv')
        panel.attrs.update(selection_method='motif', selection_trace=trace,
            reverse_complement=reverse_complement,
            motif_instances=retained_hits[retained_hits.sequence_name.isin(panel.sequence_id)].to_dict('records'))
        # Keep the public output layout identical between the two selection modes.
        if output_dir is not None:
            import shutil
            for path in (out/'panel').iterdir():
                shutil.move(str(path), out/path.name)
            (out/'panel').rmdir()
        return panel
