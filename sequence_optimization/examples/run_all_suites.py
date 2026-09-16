#!/usr/bin/env python3
"""Run all three multi-algorithm optimization suite examples.

This master script runs:
1. Optimus 5-Prime MRL (single-task)
2. DeepSTARR enhancer (multi-task)
3. Agarwal lentiMPRA (multiple independent models)

Each generates a unified candidate pool from all 13 algorithms for
subsequent QC, clustering, and solution-family discovery.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_example(script_name: str, args: argparse.Namespace) -> int:
    """Run a single example script with shared arguments."""
    script_path = Path(__file__).parent / script_name

    cmd = [
        sys.executable,
        str(script_path),
        "--device", args.device,
        "--weights-dir", args.weights_dir,
        "--initial-count", str(args.initial_count),
        "--num-candidates", str(args.num_candidates),
        "--budget", str(args.budget),
        "--seed", str(args.seed),
    ]

    if args.algorithms:
        cmd.extend(["--algorithms"] + args.algorithms)

    print(f"\n{'=' * 80}")
    print(f"Running: {script_name}")
    print(f"{'=' * 80}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="Algorithms to run (default: all 13)",
    )
    parser.add_argument(
        "--initial-count",
        type=int,
        default=100,
        help="Initial population size (default: 100)",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=50,
        help="Candidates per algorithm (default: 50)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5_000,
        help="Query budget per algorithm (default: 5000)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (cuda/cpu, default: cuda)",
    )
    parser.add_argument(
        "--weights-dir",
        default="/mnt/data/home/hexin/bioseq-models-weight-cache/0.10.0",
        help="Model weights directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["optimus", "deepstarr", "agarwal"],
        default=["optimus", "deepstarr", "agarwal"],
        help="Tasks to run (default: all three)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Algorithm Optimization Suite - Master Script")
    print("=" * 80)
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Initial population: {args.initial_count}")
    print(f"Candidates per algorithm: {args.num_candidates}")
    print(f"Budget per algorithm: {args.budget}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    scripts = {
        "optimus": "01_optimus_mrl_suite.py",
        "deepstarr": "02_deepstarr_multitask_suite.py",
        "agarwal": "03_agarwal_multimodel_suite.py",
    }

    failed = []
    for task in args.tasks:
        script = scripts[task]
        returncode = run_example(script, args)
        if returncode != 0:
            failed.append(task)

    print("\n" + "=" * 80)
    print("ALL TASKS SUMMARY")
    print("=" * 80)

    if not failed:
        print("✓ All tasks completed successfully!")
        print("\nResults saved to:")
        print("  - results/01_optimus_mrl_suite/")
        print("  - results/02_deepstarr_multitask_suite/")
        print("  - results/03_agarwal_multimodel_suite/")
    else:
        print(f"✗ {len(failed)} task(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
