#!/usr/bin/env python3
"""Multi-algorithm optimization suite for Agarwal lentiMPRA cell-type specificity.

Task: Multiple independent models (cell-type specificity)
Models: Agarwal K562, HepG2, WTC11
Region: 3'UTR (230 nt)
Objective: Maximize K562, minimize HepG2 and WTC11

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
    CombinedScorer,
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
        default=128,
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
        default="results/03_agarwal_multimodel_suite",
        help="Output directory for results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Agarwal lentiMPRA Multi-Model Multi-Algorithm Optimization Suite")
    print("=" * 80)

    # Load three cell-type models
    print(f"\nLoading models (device={args.device}, batch_size={args.model_batch_size})...")
    model_k562 = load_model("agarwal_k562", device=args.device, batch_size=args.model_batch_size)
    model_hepg2 = load_model("agarwal_hepg2", device=args.device, batch_size=args.model_batch_size)
    model_wtc11 = load_model("agarwal_wtc11", device=args.device, batch_size=args.model_batch_size)

    # Combine models
    combined = CombinedScorer(
        [model_k562, model_hepg2, model_wtc11],
        names=["K562", "HepG2", "WTC11"],
    )

    # Generate initial sequences
    print(f"Generating {args.initial_count} initial sequences (seed={args.seed})...")
    initial = random_seed_sequences(length=230, count=args.initial_count, seed=args.seed)

    # Run all 13 algorithms
    print("\nRunning all 13 optimization algorithms...")
    print("- Objective: maximize K562, minimize HepG2 and WTC11 (cell specificity)")
    print("- Default: 100 candidates per algorithm")
    print("- Random search: 100,000 samples → top 100")
    print("- Other algorithms: auto-stop at saturation plateau (0.9)")
    print("- Diversity: minimum Hamming distance = 5")

    result_df = optimize_algorithms(
        combined,
        length=230,
        initial=initial,
        objective=["maximize", "minimize", "minimize"],
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
    for idx, row in top_10.iterrows():
        k562 = row.get('activity_K562', 'N/A')
        hepg2 = row.get('activity_HepG2', 'N/A')
        wtc11 = row.get('activity_WTC11', 'N/A')
        print(f"{row['algorithm']:25s} score={row['score']:.4f}  K562={k562:.4f if k562 != 'N/A' else k562}  HepG2={hepg2:.4f if hepg2 != 'N/A' else hepg2}  WTC11={wtc11:.4f if wtc11 != 'N/A' else wtc11}")
        print(f"  {row['sequence'][:60]}...")

    # Show sample sequences from each algorithm
    print("\n" + "-" * 80)
    print("Sample sequences from each algorithm (top 3 per algorithm):")
    print("-" * 80)
    for algorithm in result_df['algorithm'].unique():
        algo_df = result_df[result_df['algorithm'] == algorithm]
        top_3 = algo_df.nlargest(3, 'score')
        print(f"\n{algorithm}:")
        for idx, row in top_3.iterrows():
            k562 = row.get('activity_K562', 'N/A')
            hepg2 = row.get('activity_HepG2', 'N/A')
            wtc11 = row.get('activity_WTC11', 'N/A')
            print(f"  score={row['score']:.4f} K562={k562:.4f if k562 != 'N/A' else k562} HepG2={hepg2:.4f if hepg2 != 'N/A' else hepg2} WTC11={wtc11:.4f if wtc11 != 'N/A' else wtc11}")
            print(f"  {row['sequence'][:60]}...")

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
