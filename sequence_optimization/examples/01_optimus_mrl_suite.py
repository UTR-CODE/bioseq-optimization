#!/usr/bin/env python3
"""Multi-algorithm optimization suite for Optimus 5-Prime MRL.

Task: Single-task maximization
Model: Optimus 5-Prime
Region: 5'UTR (50 nt)
Objective: Maximize mean ribosome load (MRL)

This script runs all 13 optimization algorithms and returns a unified
candidate pool as a pandas DataFrame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add bioseq-models to path
sys.path.insert(0, "/home/hexin/bioseq-models")

from bioseq_models import load_model

from sequence_optimization import (
    SequencePredictor,
    optimize_algorithms,
    random_seed_sequences,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-count",
        type=int,
        default=100,
        help="Initial population size (default: 100)",
    )
    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=256,
        help="Model inference batch size",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/01_optimus_mrl_suite",
        help="Output directory for results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Optimus 5-Prime MRL Multi-Algorithm Optimization Suite")
    print("=" * 80)

    # Load model
    print(f"\nLoading model (device={args.device}, batch_size={args.model_batch_size})...")
    model = load_model("optimus5prime", device=args.device, batch_size=args.model_batch_size)

    # Generate initial sequences
    print(f"Generating {args.initial_count} initial sequences (seed={args.seed})...")
    initial = random_seed_sequences(length=50, count=args.initial_count, seed=args.seed)

    # Run all 13 algorithms
    print("\nRunning all 13 optimization algorithms...")
    print("- Default: 100 candidates per algorithm")
    print("- Random search: 100,000 samples → top 100")
    print("- Other algorithms: auto-stop at saturation plateau (0.9)")
    print("- Diversity: minimum Hamming distance = 5")

    result_df = optimize_algorithms(
        model,
        length=50,
        initial=initial,
        objective=["maximize"],
        seed=args.seed,
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nTotal candidates: {len(result_df)}")
    print(f"Algorithms: {len(result_df['algorithm'].unique())}")
    print(f"\nCandidates per algorithm:")
    print(result_df.groupby('algorithm').size().to_string())

    print(f"\nBest scores per algorithm:")
    best_scores = result_df.groupby('algorithm')['score'].max().sort_values(ascending=False)
    print(best_scores.to_string())

    print(f"\nOverall best score: {result_df['score'].max():.6f}")
    print(f"Overall mean score: {result_df['score'].mean():.6f}")

    # Show top 10 candidates across all algorithms
    print("\n" + "-" * 80)
    print("Top 10 candidates (across all algorithms):")
    print("-" * 80)
    top_10 = result_df.nlargest(10, 'score')
    print(top_10[['algorithm', 'sequence', 'score']].to_string(index=False))

    # Show sample sequences from each algorithm
    print("\n" + "-" * 80)
    print("Sample sequences from each algorithm (top 3 per algorithm):")
    print("-" * 80)
    for algorithm in result_df['algorithm'].unique():
        algo_df = result_df[result_df['algorithm'] == algorithm]
        top_3 = algo_df.nlargest(3, 'score')
        print(f"\n{algorithm}:")
        for idx, row in top_3.iterrows():
            print(f"  {row['sequence']} (score: {row['score']:.6f})")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "candidates.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("✓ Optimization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

