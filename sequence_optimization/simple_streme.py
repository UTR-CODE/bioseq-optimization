"""Small STREME-inspired motif discovery for variable-length attribution seqlets.

Independent implementation: canonical k-mer seeds, PWM refinement, training-only
site erasure, and parent-group held-out enrichment. No external executables.
PWMs use [width, A/C/G/T]; scores are log2 odds against training-control bases.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import binom


@dataclass
class DiscoveryResult:
    pwms: dict
    support: pd.DataFrame
    split: pd.DataFrame
    background: list


def dinucleotide_controls(sequences, seed=42):
    """Random Euler trails preserving each fragment's dinucleotide counts."""
    rng = np.random.default_rng(seed)
    controls = []
    for sequence in sequences:
        edges = {base: [] for base in 'ACGT'}
        for a, b in zip(sequence, sequence[1:]):
            edges[a].append(b)
        for values in edges.values():
            rng.shuffle(values)
        stack, trail = [sequence[0]], []
        while stack:
            if edges[stack[-1]]:
                stack.append(edges[stack[-1]].pop())
            else:
                trail.append(stack.pop())
        controls.append(''.join(trail[::-1]))
    return controls


def encode_fragments(sequences):
    """Pad with invalid bases, which cannot participate in any scanned window."""
    sequences = [str(s).upper() for s in sequences]
    if not sequences or any(not s or set(s)-set('ACGT') for s in sequences):
        raise ValueError('Provide nonempty A/C/G/T seqlets')
    codes = np.full((len(sequences), max(map(len, sequences))), -1, dtype=np.int8)
    for i, sequence in enumerate(sequences):
        codes[i, :len(sequence)] = ['ACGT'.index(base) for base in sequence]
    return codes


@njit(cache=True, nogil=True)
def scan_best(codes, pwm, background):
    """Best window per seqlet, both strands; invalid/padded windows are skipped."""
    n, length = codes.shape
    width = len(pwm)
    logodds = np.log2(np.maximum(pwm, 1e-12)/background)
    scores = np.full(n, -np.inf)
    starts = np.full(n, -1, dtype=np.int64)
    strands = np.zeros(n, dtype=np.int8)
    for i in range(n):
        for start in range(length-width+1):
            forward, reverse, valid = 0., 0., True
            for j in range(width):
                base = codes[i, start+j]
                if base < 0:
                    valid = False
                    break
                forward += logodds[j, base]
                reverse += logodds[width-j-1, 3-base]
            if valid:
                if forward > scores[i]:
                    scores[i], starts[i], strands[i] = forward, start, 1
                if reverse > scores[i]:
                    scores[i], starts[i], strands[i] = reverse, start, -1
    # Stabilize score ties across strand summation and MEME serialization.
    return np.round(scores, 6), starts, strands


def group_max(scores, groups, count):
    result = np.full(count, -np.inf)
    np.maximum.at(result, groups, scores)
    return result


def paired_logp(positive, control, threshold):
    """One-sided exact paired test: foreground-only vs control-only groups."""
    a, b = positive >= threshold, control >= threshold
    wins, losses = int(np.sum(a & ~b)), int(np.sum(b & ~a))
    return float(binom.logsf(wins-1, wins+losses, .5)) if wins+losses else 0.


def optimal_threshold(positive, control, min_support=2):
    """Choose a threshold using training groups only, with one vote per group."""
    thresholds = np.unique(positive[np.isfinite(positive)])
    if not len(thresholds):
        return np.inf, 0.
    n = len(positive)
    above = n-np.searchsorted(np.sort(positive), thresholds, side='left')
    negative = n-np.searchsorted(np.sort(control), thresholds, side='left')
    both = n-np.searchsorted(np.sort(np.minimum(positive, control)), thresholds, side='left')
    wins, losses = above-both, negative-both
    logp = binom.logsf(wins-1, wins+losses, .5)
    logp[(above < min_support) | (wins <= losses)] = 0.
    # Highest threshold for equal enrichment: avoid admitting extra weak sites.
    best = len(logp)-1-int(np.argmin(logp[::-1]))
    return float(thresholds[best]), float(logp[best])


def _word_presence(codes, groups, width):
    windows = np.lib.stride_tricks.sliding_window_view(codes, width, axis=1)
    rows, starts = np.nonzero((windows >= 0).all(axis=2))
    values = windows[rows, starts].astype(np.int64)
    powers = 4**np.arange(width-1, -1, -1, dtype=np.int64)
    words = np.minimum(values@powers, (3-values[:, ::-1])@powers)
    return np.unique(np.rec.fromarrays([words, groups[rows]], names='word,group'))


def _seeds(positive, negative, groups, group_count, widths, per_width):
    seeds = []
    for width in widths:
        a = _word_presence(positive, groups, width)
        b = _word_presence(negative, groups, width)
        words = np.unique(np.concatenate([a['word'], b['word']]))
        pos = np.zeros(len(words), dtype=int)
        neg, both = pos.copy(), pos.copy()
        for pairs, target in [(a, pos), (b, neg), (np.intersect1d(a, b), both)]:
            keys, numbers = np.unique(pairs['word'], return_counts=True)
            target[np.searchsorted(words, keys)] = numbers
        wins, losses = pos-both, neg-both
        logp = binom.logsf(wins-1, wins+losses, .5)
        valid = np.flatnonzero((pos >= 2) & (wins > losses))
        order = valid[np.lexsort((words[valid], -pos[valid], logp[valid]))][:per_width]
        for index in order:
            word = int(words[index])
            digits = np.array([(word//int(4**j)) % 4 for j in range(width-1, -1, -1)])
            pwm = np.full((width, 4), .1)
            pwm[np.arange(width), digits] = .7
            seeds.append((''.join(np.array(list('ACGT'))[digits]), pwm))
    return seeds


def _refine(codes, controls, groups, group_count, pwm, background, iterations):
    best, previous = None, None
    for _ in range(iterations):
        scores, starts, strands = scan_best(codes, pwm, background)
        negative = scan_best(controls, pwm, background)[0]
        threshold, logp = optimal_threshold(group_max(scores, groups, group_count),
            group_max(negative, groups, group_count))
        if best is None or logp < best[0]:
            best = (logp, threshold, pwm.copy())
        passing = np.flatnonzero(scores >= threshold)
        if len(passing) < 2:
            break
        # One best-aligned window per independent group estimates the PWM.
        ranked = passing[np.argsort(-scores[passing], kind='stable')]
        _, first = np.unique(groups[ranked], return_index=True)
        sites = []
        for row in ranked[first]:
            site = codes[row, starts[row]:starts[row]+len(pwm)].copy()
            sites.append(site if strands[row] == 1 else 3-site[::-1])
        sites = np.asarray(sites)
        if previous is not None and np.array_equal(sites, previous):
            break
        previous = sites.copy()
        counts = np.tile(background*.5, (len(pwm), 1))
        for base in range(4):
            counts[:, base] += (sites == base).sum(axis=0)
        pwm = counts/counts.sum(axis=1, keepdims=True)
    return best


def _erase(codes, pwm, background, threshold):
    # Repeated best-site masking also handles several instances in one fragment.
    while True:
        scores, starts, _ = scan_best(codes, pwm, background)
        rows = np.flatnonzero(np.isfinite(scores) & (scores >= threshold))
        if not len(rows):
            return
        for row in rows:
            codes[row, starts[row]:starts[row]+len(pwm)] = -1


def discover_seqlet_motifs(seqlets, *, controls=None, groups=None,
                           min_width=6, max_width=15, max_motifs=15,
                           min_seqlets=10, min_parents=3, holdout_fraction=.2,
                           validation_mask=None, per_width=3, iterations=10,
                           alpha=.05, seed=42):
    """Discover variable-width motifs; return candidates and explicit retention flags.

    Input columns: dna, sequence_id; optional seqlet_id. Independent statistical
    units default to parent sequence IDs; groups can provide sequence families.
    Training determines all motif candidates, thresholds and erasures. Held-out
    data never determine the next candidate or early stopping. Reported validation
    p-values use fixed thresholds and max_motifs Bonferroni correction. Counts and
    support are on the original fragments, retaining informative flank columns.
    """
    frame = seqlets.reset_index(drop=True).copy()
    if not {'dna', 'sequence_id'} <= set(frame):
        raise ValueError('Seqlets require dna and sequence_id columns')
    if not 1 <= min_width <= max_width <= 30 or max_motifs < 1 or per_width < 1 or iterations < 1:
        raise ValueError('Invalid discovery widths or search counts')
    if min_seqlets < 1 or min_parents < 1 or not 0 < alpha < 1 or not 0 < holdout_fraction < 1:
        raise ValueError('Invalid support, significance, or holdout settings')
    if 'seqlet_id' not in frame:
        frame['seqlet_id'] = [f'Q{i+1:06}' for i in range(len(frame))]
    foreground = frame.dna.astype(str).str.upper().tolist()
    original = encode_fragments(foreground)
    negatives = dinucleotide_controls(foreground, seed) if controls is None else list(controls)
    if len(negatives) != len(foreground) or any(len(a) != len(b) for a, b in zip(foreground, negatives)):
        raise ValueError('Provide one equal-length control for each seqlet')
    control_codes = encode_fragments(negatives)
    # Parent rows are never split; explicit family groups must also keep parents whole.
    labels = frame.sequence_id.astype(str).to_numpy() if groups is None else np.asarray(groups).astype(str)
    if labels.shape != (len(frame),):
        raise ValueError('Provide one independent-group label per seqlet')
    parent_groups = pd.DataFrame(dict(parent=frame.sequence_id, group=labels))
    if parent_groups.groupby('parent').group.nunique().max() > 1:
        raise ValueError('Each parent sequence must belong to one discovery group')
    unique_groups, group_ids = np.unique(labels, return_inverse=True)
    count = len(unique_groups)
    if count < 10:
        raise ValueError('Held-out motif discovery requires at least 10 independent groups')
    if validation_mask is None:
        permutation = np.random.default_rng(seed).permutation(count)
        hold = permutation[:max(3, int(np.ceil(count*holdout_fraction))) ]
        validation_mask = np.isin(group_ids, hold)
    else:
        validation_mask = np.asarray(validation_mask, dtype=bool)
        if validation_mask.shape != (len(frame),):
            raise ValueError('validation_mask must have one value per seqlet')
    if set(group_ids[validation_mask]) & set(group_ids[~validation_mask]):
        raise ValueError('Independent groups must not cross training/validation splits')
    if min(len(set(group_ids[validation_mask])), len(set(group_ids[~validation_mask]))) < 3:
        raise ValueError('Require at least three independent groups in each split')
    training = ~validation_mask
    positive, negative = original[training].copy(), control_codes[training].copy()
    train_ids = np.unique(group_ids[training])
    train_groups = np.searchsorted(train_ids, group_ids[training])
    validation_ids = np.unique(group_ids[validation_mask])
    val_groups = np.searchsorted(validation_ids, group_ids[validation_mask])
    frequencies = np.bincount(negative[negative >= 0], minlength=4)+1.
    background = (frequencies+frequencies[::-1])/(2*frequencies.sum())
    widths = range(min_width, min(max_width, original.shape[1])+1)
    rows, pwms = [], {}
    for round_index in range(max_motifs):
        best = None
        for word, pwm in _seeds(positive, negative, train_groups, len(train_ids), widths, per_width):
            result = _refine(positive, negative, train_groups, len(train_ids), pwm, background, iterations)
            logp, threshold, pwm = result
            information = float(np.sum(pwm*np.log2(np.maximum(pwm, 1e-12)/background)))
            key = (logp, -information)
            if best is None or key < best[0]:
                best = (key, threshold, pwm, word)
        if best is None or best[0][0] >= np.log(alpha):
            break  # Training-only stopping: held-out results never drive search.
        _, search_threshold, pwm, word = best
        motif_id = f'M{round_index+1:03}'
        pwms[motif_id] = pwm
        a = scan_best(original, pwm, background)[0]
        b = scan_best(control_codes, pwm, background)[0]
        threshold, train_logp = optimal_threshold(
            group_max(a[training], train_groups, len(train_ids)),
            group_max(b[training], train_groups, len(train_ids)))
        support = np.isfinite(a) & (a >= threshold)
        val_a = group_max(a[validation_mask], val_groups, len(validation_ids))
        val_b = group_max(b[validation_mask], val_groups, len(validation_ids))
        logp = paired_logp(val_a, val_b, threshold)
        adjusted = min(1., float(np.exp(logp)*max_motifs))
        parents = frame.loc[support, 'sequence_id'].nunique()
        rows.append(dict(motif_id=motif_id, seed_word=word,
            consensus=''.join(np.array(list('ACGT'))[pwm.argmax(1)]), width=len(pwm),
            threshold_bits=threshold, train_logp=train_logp,
            search_threshold_bits=search_threshold, search_train_logp=best[0][0],
            heldout_logp=logp, heldout_p_adjusted=adjusted,
            heldout_positive_groups=int(np.sum(val_a >= threshold)),
            heldout_control_groups=int(np.sum(val_b >= threshold)),
            seqlets=int(support.sum()), parents=int(parents),
            retained=bool(adjusted <= alpha and support.sum() >= min_seqlets and parents >= min_parents)))
        _erase(positive, pwm, background, search_threshold)
        _erase(negative, pwm, background, search_threshold)
    columns = ['motif_id', 'seed_word', 'consensus', 'width', 'threshold_bits', 'train_logp',
        'search_threshold_bits', 'search_train_logp', 'heldout_logp', 'heldout_p_adjusted', 'heldout_positive_groups',
        'heldout_control_groups', 'seqlets', 'parents', 'retained']
    split = frame[['seqlet_id', 'sequence_id']].copy()
    split['discovery_group'] = labels
    split['split'] = np.where(validation_mask, 'heldout', 'train')
    return DiscoveryResult(pwms, pd.DataFrame(rows, columns=columns), split, negatives)
