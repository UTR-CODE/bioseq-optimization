#!/usr/bin/env python3
"""Demo script showing CandidatePool analysis capabilities.

This script demonstrates how to use the CandidatePool class to analyze
unified candidate results from multiple optimization algorithms.
"""

import sys
sys.path.insert(0, "/home/hexin/bioseq-models")

from pathlib import Path

import pandas as pd
from bioseq_models import load_model

from sequence_optimization import (
    CandidatePool,
    optimize_algorithms,
    random_seed_sequences,
)


def main():
    """Run a small optimization suite and demonstrate pool analysis."""
    print("=" * 80)
    print("CandidatePool Analysis Demo")
    print("=" * 80)

    # Load model
    print("\nLoading Optimus 5-Prime model...")
    model = load_model("optimus5prime", device="cpu", batch_size=256)

    # Generate initial sequences
    print("Generating initial sequences...")
    initial = random_seed_sequences(length=50, count=20, seed=42)

    # Run subset of algorithms for demo
    print("\nRunning 4 algorithms...")
    algorithms = ["genetic_algorithm", "basesearch", "simulated_annealing", "ism_greedy"]

    suite = optimize_algorithms(
        model,
        algorithms=algorithms,
        length=50,
        initial=initial,
        objective=["maximize"],
        num_candidates=10,
        budget=500,
        seed=42,
    )

    print("\nAlgorithm summary:")
    print(suite.summary_frame().to_string(index=False))

    # Create unified candidate pool
    print("\n" + "=" * 80)
    print("CANDIDATE POOL ANALYSIS")
    print("=" * 80)

    pool = CandidatePool.from_suite_result(suite)
    print(f"\n✓ Created candidate pool: {pool}")

    # Show all candidates
    print("\n" + "-" * 80)
    print("All candidates (before deduplication):")
    print("-" * 80)
    df = pool.to_frame()
    print(df[['algorithm', 'rank', 'sequence', 'score']].to_string(index=False))
    print(f"\nTotal candidates: {len(pool)}")

    # Analyze convergence
    print("\n" + "-" * 80)
    print("Convergence Analysis (sequences found by multiple algorithms):")
    print("-" * 80)
    convergence = pool.get_convergence_summary()
    multi_algo = convergence[convergence['algorithm_count'] > 1]

    if len(multi_algo) > 0:
        print(multi_algo[['sequence', 'score', 'algorithm_count', 'algorithms', 'mean_rank']].to_string(index=False))
        print(f"\n✓ {len(multi_algo)} sequences found by multiple algorithms")
    else:
        print("No sequences were found by multiple algorithms in this small demo.")
        print("(Increase budget and num_candidates to see convergence)")

    # Deduplicate
    print("\n" + "-" * 80)
    print("Deduplication (keep best score for each sequence):")
    print("-" * 80)
    unique_pool = pool.deduplicate(keep='best')
    print(f"Before: {len(pool)} candidates")
    print(f"After:  {len(unique_pool)} unique candidates")
    print(f"Removed: {len(pool) - len(unique_pool)} duplicates")

    # Top candidates across all algorithms
    print("\n" + "-" * 80)
    print("Top 5 candidates (across all algorithms):")
    print("-" * 80)
    top_5 = unique_pool.top_n_by_score(5)
    top_df = top_5.to_frame()
    print(top_df[['algorithm', 'sequence', 'score']].to_string(index=False))

    # Algorithm comparison
    print("\n" + "-" * 80)
    print("Best score per algorithm:")
    print("-" * 80)
    best_by_algo = df.groupby('algorithm').agg({
        'score': ['max', 'mean', 'min'],
        'sequence': 'count'
    }).round(3)
    best_by_algo.columns = ['best', 'mean', 'worst', 'count']
    print(best_by_algo.to_string())

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Ran {len(algorithms)} algorithms")
    print(f"✓ Generated {len(pool)} total candidates")
    print(f"✓ Found {len(unique_pool)} unique sequences")
    print(f"✓ Best score overall: {df['score'].max():.3f}")
    print(f"✓ Algorithm with highest best: {df.loc[df['score'].idxmax(), 'algorithm']}")

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
