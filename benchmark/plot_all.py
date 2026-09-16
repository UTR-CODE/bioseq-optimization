"""Plot every task panel individually and assemble the Fig. 2 overview.

Usage:
    python plot_all.py
"""
import sys
import os

sys.path.insert(0, '/home/hexin/bioseq-models')
sys.path.insert(0, '/mnt/data/home/hexin/MPRA_benchmark')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import PLOTS_DIR
from plot_style import collect, draw_panel, legend_handles
from plot_task import plot_individual


def plot_overview(tasks):
    ncols = 3
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.0, nrows * 3.6))
    axes_flat = axes.flatten()

    any_baseline = False
    for index, task in enumerate(tasks):
        means, bests, statuses, colors, baseline = collect(task['id'])
        any_baseline = any_baseline or baseline is not None
        ax = axes_flat[index]
        draw_panel(ax, task, means, bests, statuses, colors, baseline, compact=True)
        ax.set_title(task['label'], fontsize=8, fontweight='bold',
                     loc='left', pad=5)
        if index % ncols == 0:
            ax.set_ylabel('Optimized score', fontsize=8)

    for empty in range(len(tasks), len(axes_flat)):
        axes_flat[empty].set_visible(False)

    fig.legend(handles=legend_handles(any_baseline), fontsize=8.5,
               frameon=False, loc='upper center', ncol=6,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        'Fig. 2  Algorithm benchmark across nine regulatory sequence design tasks',
        fontsize=11.5, fontweight='bold', y=1.035,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=3.4, w_pad=2.2)

    out_path = os.path.join(PLOTS_DIR, 'fig2_overview.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved overview: {out_path}')


def main():
    from tasks import TASKS
    print('Plotting individual task panels ...')
    for task in TASKS:
        plot_individual(task)
    print('\nAssembling Fig. 2 overview ...')
    plot_overview(TASKS)


if __name__ == '__main__':
    main()
