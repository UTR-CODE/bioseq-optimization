#!/usr/bin/env python3
"""Multi-algorithm optimization suite for Agarwal lentiMPRA cell specificity.

Task: Multiple independent models
Models: Agarwal K562, HepG2, WTC11
Region: Enhancer (230 nt)
Objective: Maximize K562 specificity against HepG2 and WTC11
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
    CombinedScorer,
    DIFFERENTIABLE_ALGORITHMS,
    SequencePredictor,
    available_algorithms,
    optimize_algorithms,
    random_seed_sequences,
)


class AgarwalPredictor(SequencePredictor):
    """Custom predictor for Agarwal models using AGCT channel order."""

    def prepare_input(self, values):
        # Agarwal uses AGCT order instead of ACGT
        return values[..., [0, 2, 1, 3]].transpose(1, 2)


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
        default="results/03_agarwal_multimodel_suite",
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
    print("Multi-Algorithm Optimization Suite: Agarwal lentiMPRA")
    print("=" * 80)
    print(f"Task: Multiple independent models")
    print(f"Region: Enhancer (230 nt)")
    print(f"Objective: Maximize K562, Minimize HepG2 and WTC11")
    print(f"Algorithms: {len(args.algorithms)}")
    print(f"Initial population: {args.initial_count}")
    print(f"Candidates per algorithm: {args.num_candidates}")
    print(f"Budget per algorithm: {args.budget}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load three independent models
    print("\nLoading Agarwal models (K562, HepG2, WTC11)...")
    k562 = load_model(
        "agarwal_k562",
        device=args.device,
        batch_size=args.model_batch_size,
        weights_dir=args.weights_dir,
    )
    hepg2 = load_model(
        "agarwal_hepg2",
        device=args.device,
        batch_size=args.model_batch_size,
        weights_dir=args.weights_dir,
    )
    wtc11 = load_model(
        "agarwal_wtc11",
        device=args.device,
        batch_size=args.model_batch_size,
        weights_dir=args.weights_dir,
    )

    # Combine models for black-box optimization
    model = CombinedScorer(
        [k562, hepg2, wtc11],
        names=["K562", "HepG2", "WTC11"],
    )
    print(f"Combined scorer created with tasks: {model.names}")

    # Prepare algorithms
    selected = tuple(args.algorithms)
    differentiable_model = None
    if set(selected) & DIFFERENTIABLE_ALGORITHMS:
        print("\nWrapping models for gradient algorithms...")
        k562_gradient = AgarwalPredictor(
            k562.model,
            device=args.device,
            task_names=k562.task_names,
            batch_size=args.model_batch_size,
        )
        hepg2_gradient = AgarwalPredictor(
            hepg2.model,
            device=args.device,
            task_names=hepg2.task_names,
            batch_size=args.model_batch_size,
        )
        wtc11_gradient = AgarwalPredictor(
            wtc11.model,
            device=args.device,
            task_names=wtc11.task_names,
            batch_size=args.model_batch_size,
        )
        differentiable_model = CombinedScorer(
            [k562_gradient, hepg2_gradient, wtc11_gradient],
            names=["K562", "HepG2", "WTC11"],
        )

    # Generate initial population
    print(f"\nGenerating {args.initial_count} initial sequences (seed={args.seed})...")
    initial = random_seed_sequences(
        length=230,
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
        length=230,
        initial=initial,
        objective=["maximize", "minimize", "minimize"],
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
        "task": "Agarwal lentiMPRA cell specificity optimization",
        "models": ["agarwal_k562", "agarwal_hepg2", "agarwal_wtc11"],
        "region": "enhancer",
        "length": 230,
        "objective": ["maximize", "minimize", "minimize"],
        "task_names": ["K562", "HepG2", "WTC11"],
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

    # Display query counts per model
    print("\n" + "=" * 80)
    print("MODEL QUERY COUNTS (shared cache across algorithms)")
    print("=" * 80)
    print(f"K562:  {model.model_query_counts.get('K562', 0)}")
    print(f"HepG2: {model.model_query_counts.get('HepG2', 0)}")
    print(f"WTC11: {model.model_query_counts.get('WTC11', 0)}")

    print("\n" + "=" * 80)
    print("TOP CANDIDATES ACROSS ALL ALGORITHMS")
    print("=" * 80)
    all_candidates = suite.to_frame()
    top_20 = all_candidates.nlargest(20, 'score')[
        ['algorithm', 'rank', 'sequence', 'score',
         'activity_K562', 'activity_HepG2', 'activity_WTC11',
         'query_count', 'elapsed_seconds']
    ]
    print(top_20.to_string(index=False))

    print(f"\n✓ Complete! Total unique candidates: {len(all_candidates)}")
    print(f"✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
