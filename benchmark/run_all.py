"""Launch all 9 benchmark tasks sequentially.

Usage:
    python run_all.py [--tasks 0 1 2 ...]   # subset of task indices
    python run_all.py                        # all 9 tasks
"""
import sys
import os
import subprocess
import argparse

sys.path.insert(0, '/home/hexin/bioseq-models')
sys.path.insert(0, '/mnt/data/home/hexin/MPRA_benchmark')

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', type=int,
                        default=list(range(9)),
                        help='Task indices to run (0-8)')
    args = parser.parse_args()

    for idx in args.tasks:
        print(f'\n{"#"*60}')
        print(f'# Launching task {idx}')
        print(f'{"#"*60}')
        ret = subprocess.run(
            [sys.executable, os.path.join(HERE, 'run_task.py'), str(idx)],
            check=False,
        )
        if ret.returncode != 0:
            print(f'WARNING: task {idx} exited with code {ret.returncode}')

    print('\nAll tasks finished.')


if __name__ == '__main__':
    main()
