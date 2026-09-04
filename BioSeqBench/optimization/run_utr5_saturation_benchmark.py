"""CLI for optimizer-specific saturation and q90 Optimus 5'UTR design."""

from __future__ import annotations

import argparse

from BioSeqBench.optimization.utr5_saturation_benchmark import SaturationConfig, run_saturation_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", nargs="+", type=float, default=None)
    parser.add_argument("--num-replicates", type=int, default=5)
    parser.add_argument("--minimum-queries", type=int, default=20_000)
    parser.add_argument("--plateau-patience-queries", type=int, default=50_000)
    parser.add_argument("--maximum-queries", type=int, default=100_000_000)
    parser.add_argument("--max-runtime-seconds", type=float, default=900.0)
    parser.add_argument("--material-improvement", type=float, default=1e-4)
    parser.add_argument("--saturation-fraction", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="BioSeqBench/results/utr5_optimization_saturation90_all60")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    defaults = SaturationConfig()
    config = SaturationConfig(
        train_fractions=tuple(args.fractions) if args.fractions else defaults.train_fractions,
        num_replicates=args.num_replicates,
        minimum_queries=args.minimum_queries,
        plateau_patience_queries=args.plateau_patience_queries,
        maximum_queries=args.maximum_queries,
        max_runtime_seconds=args.max_runtime_seconds,
        material_improvement=args.material_improvement,
        saturation_fraction=args.saturation_fraction,
        base_seed=args.seed,
    )
    summary = run_saturation_benchmark(output_dir=args.output_dir, device=args.device, config=config, resume=args.resume)
    print(summary[["weak_model_fraction", "replicate", "algorithm", "q90_query_count", "mean_strong_score", "best_strong_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
