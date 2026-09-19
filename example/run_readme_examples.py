#!/usr/bin/env python3
"""Run the three README examples verbatim and save each candidate table.

Outputs land in this directory; it is gitignored, so results stay local.
"""

from __future__ import annotations

import pathlib
import sys
import time

OUT = pathlib.Path(__file__).resolve().parent


def example_1():
    from bioseq_models import load_model
    from sequence_optimization import generate_designs, random_seed_sequences

    # Load model
    model = load_model("optimus5prime", device="cuda", batch_size=256)

    # Generate initial sequences
    initial = random_seed_sequences(length=50, count=100, seed=42)

    # Run all 13 algorithms
    result_df = generate_designs(
        model,
        length=50,
        initial=initial,
        objective=["maximize"],
        seed=42,
    )

    print(result_df.head(20))
    print(f"\nTotal candidates: {len(result_df)}")
    print(f"Best score: {result_df['score'].max():.4f}")
    return result_df


def example_2():
    from bioseq_models import load_model
    from sequence_optimization import generate_designs, random_seed_sequences

    # Load model
    model = load_model("deepstarr", device="cuda", batch_size=128)

    # Generate initial sequences
    initial = random_seed_sequences(length=249, count=100, seed=42)

    # Run all 13 algorithms
    # Objective: minimize developmental activity, maximize housekeeping activity
    result_df = generate_designs(
        model,
        length=249,
        initial=initial,
        objective=["minimize", "maximize"],
        seed=42,
    )

    print(
        result_df[
            ["algorithm", "score", "activity_developmental", "activity_housekeeping"]
        ].head()
    )
    return result_df


def example_3():
    from bioseq_models import load_model
    from sequence_optimization import (
        CombinedScorer,
        generate_designs,
        random_seed_sequences,
    )

    # Load three cell-type models
    model_k562 = load_model("agarwal_k562", device="cuda", batch_size=128)
    model_hepg2 = load_model("agarwal_hepg2", device="cuda", batch_size=128)
    model_wtc11 = load_model("agarwal_wtc11", device="cuda", batch_size=128)

    # Combine models
    combined = CombinedScorer(
        [model_k562, model_hepg2, model_wtc11],
        names=["K562", "HepG2", "WTC11"],
    )

    # Generate initial sequences
    initial = random_seed_sequences(length=230, count=100, seed=42)

    # Run all 13 algorithms
    # Objective: maximize K562, minimize HepG2 and WTC11 (cell-type specificity)
    result_df = generate_designs(
        combined,
        length=230,
        initial=initial,
        objective=["maximize", "minimize", "minimize"],
        seed=42,
    )

    print(
        result_df[
            ["algorithm", "score", "activity_K562", "activity_HepG2", "activity_WTC11"]
        ].head()
    )
    return result_df


EXAMPLES = {
    "01_optimus_mrl": example_1,
    "02_deepstarr_multitask": example_2,
    "03_agarwal_multimodel": example_3,
}


def main() -> int:
    requested = sys.argv[1:] or list(EXAMPLES)
    failures = []
    for name in requested:
        print("=" * 78)
        print(f"Example: {name}")
        print("=" * 78)
        started = time.time()
        try:
            frame = EXAMPLES[name]()
        except Exception as error:  # keep going so one failure does not hide others
            failures.append((name, error))
            print(f"FAILED: {type(error).__name__}: {error}")
            continue
        path = OUT / f"{name}_candidates.csv"
        frame.to_csv(path, index=False)
        print(f"\nelapsed {time.time() - started:.1f}s")
        print(f"candidates {len(frame)} from {frame['algorithm'].nunique()} algorithms")
        print(f"saved {path}")

    print("=" * 78)
    for name, error in failures:
        print(f"FAILED {name}: {type(error).__name__}: {error}")
    print("all examples passed" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
