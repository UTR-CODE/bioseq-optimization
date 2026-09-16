"""Plot one Fig. 2 panel for a single task.

Usage:
    python plot_task.py <task_index>
"""
import sys
import os

sys.path.insert(0, '/home/hexin/bioseq-models')
sys.path.insert(0, '/mnt/data/home/hexin/MPRA_benchmark')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import PLOTS_DIR
from plot_style import collect, draw_panel, legend_handles, subtitle


def plot_individual(task):
    task_id = task['id']
    means, bests, statuses, colors, baseline = collect(task_id)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    draw_panel(ax, task, means, bests, statuses, colors, baseline)

    ax.set_ylabel('Optimized score', fontsize=9)
    ax.set_title(task['label'].replace('\n', '  ·  '),
                 fontsize=10.5, fontweight='bold', pad=10)
    ax.text(0.0, 1.015, subtitle(task_id, task), transform=ax.transAxes,
            fontsize=7.5, color='#5A5A62', va='bottom')
    ax.legend(handles=legend_handles(baseline is not None),
              fontsize=7.5, frameon=False, ncol=3,
              loc='upper left', bbox_to_anchor=(0.0, -0.28))

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f'{task_id}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out_path}')
    return out_path


def main():
    if len(sys.argv) < 2:
        print('Usage: python plot_task.py <task_index>')
        sys.exit(1)
    from tasks import TASKS
    task = TASKS[int(sys.argv[1])]
    print(f"Plotting {task['id']} ...")
    plot_individual(task)


if __name__ == '__main__':
    main()
