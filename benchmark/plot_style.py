"""Shared panel-drawing logic for Fig. 2 so individual and overview plots match."""
import os
import pickle

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from config import (
    RESULTS_DIR, ALGORITHMS, ALGO_LABELS,
    ALGO_CATEGORY, CATEGORY_COLORS, NUM_RETURN,
)

CATEGORY_ORDER = ['discrete', 'evolutionary', 'distributional', 'gradient']
PANEL_BG = '#FFFFFF'
GRID_COLOR = '#E4E4E4'
SPINE_COLOR = '#BFBFBF'
BASELINE_COLOR = '#666666'


def load_result(task_id, algo):
    path = os.path.join(RESULTS_DIR, f'{task_id}__{algo}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def collect(task_id):
    """Return per-algorithm summary arrays plus the shared random-start baseline.

    ``mean_scores`` is the mean of the returned top-N candidates and drives the
    bar height; ``best_scores`` is the single best candidate. ``baseline`` is the
    top-N mean of the initial 1,000 random sequences, read from the first
    trajectory snapshot, and is identical across algorithms by construction.
    """
    means, bests, statuses, colors = [], [], [], []
    baselines = []
    for algo in ALGORITHMS:
        record = load_result(task_id, algo)
        colors.append(CATEGORY_COLORS[ALGO_CATEGORY[algo]])
        if record is None:
            means.append(np.nan)
            bests.append(np.nan)
            statuses.append('missing')
            continue
        if record.get('status') != 'ok':
            means.append(np.nan)
            bests.append(np.nan)
            statuses.append(record.get('status', 'unknown'))
            continue
        scores = np.asarray(record['scores'], dtype=float)
        means.append(float(scores.mean()))
        bests.append(float(scores.max()))
        statuses.append('ok')
        trajectory = record.get('trajectory') or []
        if trajectory and 'top_mean_score' in trajectory[0]:
            baselines.append(float(trajectory[0]['top_mean_score']))
    baseline = float(np.median(baselines)) if baselines else None
    return (np.asarray(means), np.asarray(bests), statuses, colors, baseline)


def draw_panel(ax, task, means, bests, statuses, colors, baseline, *, compact=False):
    """Draw one task panel: bars = top-N mean, diamonds = best candidate."""
    label_size = 7 if compact else 8.5
    n = len(ALGORITHMS)
    x = np.arange(n)

    ax.set_facecolor(PANEL_BG)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.6)

    ax.bar(x, means, 0.62, color=colors, edgecolor='white', linewidth=0.6, zorder=2)

    finite = np.isfinite(means)
    if finite.any():
        ax.scatter(
            x[finite], bests[finite],
            marker='D', s=16 if compact else 22,
            facecolor='white', edgecolor='#33333A',
            linewidth=0.9, zorder=4,
        )

    if baseline is not None:
        ax.axhline(
            baseline, color=BASELINE_COLOR, linewidth=1.0,
            linestyle=(0, (4, 3)), zorder=3,
        )

    # Zoom to the data range; a 0-anchored axis hides the differences that matter.
    values = [v for v in np.concatenate([means, bests]) if np.isfinite(v)]
    if baseline is not None:
        values.append(baseline)
    if values:
        low, high = min(values), max(values)
        span = high - low
        pad = span * 0.15 if span > 0 else max(abs(high) * 0.05, 0.05)
        ax.set_ylim(low - pad, high + pad * 1.6)

    # Failed or skipped algorithms get a marker on the floor of the axis.
    floor = ax.get_ylim()[0]
    for index, status in enumerate(statuses):
        if status != 'ok':
            ax.plot(
                x[index], floor, marker='x', markersize=4.5,
                color='#9A9AA2', clip_on=False, zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [ALGO_LABELS[a] for a in ALGORITHMS],
        rotation=38, ha='right', fontsize=label_size,
    )
    ax.set_xlim(-0.7, n - 0.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SPINE_COLOR)
    ax.tick_params(axis='both', labelsize=label_size, color=SPINE_COLOR)


def legend_handles(include_baseline=True):
    handles = [
        mpatches.Patch(color=CATEGORY_COLORS[category], label=category.capitalize())
        for category in CATEGORY_ORDER
    ]
    handles.append(Line2D(
        [], [], marker='D', linestyle='none', markerfacecolor='white',
        markeredgecolor='#33333A', markersize=5, label='Best candidate',
    ))
    if include_baseline:
        handles.append(Line2D(
            [], [], color=BASELINE_COLOR, linewidth=1.0,
            linestyle=(0, (4, 3)), label='Random start (1,000 seqs)',
        ))
    return handles


def subtitle(task_id, task):
    records = [load_result(task_id, a) for a in ALGORITHMS]
    ok = [r for r in records if r and r.get('status') == 'ok']
    if not ok:
        return task['seq_type']
    median_queries = int(np.median([r['query_count'] for r in ok]))
    return (f"{task['seq_type']}   ·   bar = top-{NUM_RETURN} mean   ·   "
            f"median queries: {median_queries:,}")
