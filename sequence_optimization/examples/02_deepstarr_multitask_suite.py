#!/usr/bin/env python3
"""Multi-algorithm optimization suite for DeepSTARR enhancer design.

Task: Multi-task optimization
Model: DeepSTARR
Region: Enhancer (249 nt)
Objective: Minimize developmental activity, maximize housekeeping activity
Algorithms: All 13 available optimizers (9 black-box + 4 gradient)

This script generates a unified candidate pool from all algorithms for
subsequent QC, clustering, and solution-family discovery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add bioseq-models to path
sys.path.insert(0, "/home/hexin/bioseq-models")

from bioseq_models import load_model

from sequence_optimization import (
    DIFFERENTIABLE_ALGORITHMS,
    SequencePredictor,
    available_algorithms,
    optimize_algorithms,
    random_seed_sequences,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=available_algorithms(),
        default=list(available_algorithms()),
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
        "--model-batch-size",
        type=int,
        default=4_096,
        help="Model inference batch size",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (cuda/cpu)",
    )
    parser.add_argument(
        "--weights-dir",
        default="/mnt/data/home/hexin/bioseq-models-weight-cache/0.10.0",
        help="Model weights directory",
    )
    parser.add_argument(
        "--output-dir",
        default="results/02_deepstarr_multitask_suite",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Algorithm Optimization Suite: DeepSTARR Enhancer")
    print("=" * 80)
    print(f"Task: Multi-task optimization")
    print(f"Region: Enhancer (249 nt)")
    print(f"Objective: Minimize developmental, Maximize housekeeping")
    print(f"Algorithms: {len(args.algorithms)}")
    print(f"Initial population: {args.initial_count}")
    print(f"Candidates per algorithm: {args.num_candidates}")
    print(f"Budget per algorithm: {args.budget}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load model
    print("\nLoading DeepSTARR model...")
    model = load_model(
        "deepstarr",
        device=args.device,
        batch_size=args.model_batch_size,
        weights_dir=args.weights_dir,
    )
    print(f"Model loaded. Task names: {model.task_names}")

    # Prepare algorithms
    selected = tuple(args.algorithms)
    differentiable_model = None
    if set(selected) & DIFFERENTIABLE_ALGORITHMS:
        print("\nWrapping model for gradient algorithms...")
        differentiable_model = SequencePredictor(
            model.model,
            device=model.device,
            task_names=model.task_names,
            batch_size=args.model_batch_size,
        )

    # Generate initial population
    print(f"\nGenerating {args.initial_count} initial sequences (seed={args.seed})...")
    initial = random_seed_sequences(
        length=249,
        count=args.initial_count,
        seed=args.seed,
    )

    # Run optimization suite
    print(f"\nRunning {len(selected)} algorithms...")
    print("This may take several minutes depending on device and budget...")
    suite = optimize_algorithms(
        model,
        algorithms=selected,
        differentiable_model=differentiable_model,
        length=249,
        initial=initial,
        objective=["minimize", "maximize"],
        num_candidates=args.num_candidates,
        budget=args.budget,
        seed=args.seed,
    )

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {output_path}...")
    paths = suite.save_csv(output_path)

    # Save configuration
    config = {
        "task": "DeepSTARR enhancer multi-task optimization",
        "model": "deepstarr",
        "region": "enhancer",
        "length": 249,
        "objective": ["minimize", "maximize"],
        "task_names": list(model.task_names),
        "algorithms": list(selected),
        "initial_count": args.initial_count,
        "num_candidates": args.num_candidates,
        "budget": args.budget,
        "model_batch_size": args.model_batch_size,
        "device": args.device,
        "seed": args.seed,
        "weights_dir": args.weights_dir,
    }

    config_path = output_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    print(f"\nSaved files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"  config: {config_path}")

    # Display summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    summary = suite.summary_frame()
    print(summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP CANDIDATES ACROSS ALL ALGORITHMS")
    print("=" * 80)
    all_candidates = suite.to_frame()
    top_20 = all_candidates.nlargest(20, 'score')[
        ['algorithm', 'rank', 'sequence', 'score',
         'activity_developmental', 'activity_housekeeping',
         'query_count', 'elapsed_seconds']
    ]
    print(top_20.to_string(index=False))

    print(f"\n✓ Complete! Total unique candidates: {len(all_candidates)}")
    print(f"✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
