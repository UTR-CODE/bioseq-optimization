"""CLI for the three-model, nine-algorithm TFBind8 optimization benchmark."""

from __future__ import annotations

import argparse

from BioSeqBench.optimization.tfbind8_benchmark import (
    ALGORITHM_NAMES,
    TFBind8OptimizationConfig,
    run_tfbind8_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.01, 0.50, 0.95],
        help="Lower-tail weak-model fractions (default: 0.01 0.50 0.95)",
    )
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_NAMES, default=list(ALGORITHM_NAMES))
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--query-budget", type=int, default=10_000)
    parser.add_argument("--num-replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--checkpoint-root", default="BioSeqBench/results/tfbind_weak_cnns/tfbind8")
    parser.add_argument("--output-dir", default="BioSeqBench/results/tfbind8_optimization_benchmark")
    parser.add_argument("--raw-path")
    parser.add_argument("--device")
    parser.add_argument("--cbas-rounds", type=int, default=10)
    parser.add_argument("--cbas-pretrain-epochs", type=int, default=20)
    parser.add_argument("--cbas-update-epochs", type=int, default=10)
    parser.add_argument("--den-seed-pretrain-epochs", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = TFBind8OptimizationConfig(
        train_fractions=tuple(args.fractions),
        algorithms=tuple(args.algorithms),
        num_candidates=args.num_candidates,
        query_budget=args.query_budget,
        base_seed=args.seed,
        top_visible_initializations=args.num_candidates // 2,
        random_initializations=args.num_candidates - args.num_candidates // 2,
        cbas_rounds=args.cbas_rounds,
        cbas_pretrain_epochs=args.cbas_pretrain_epochs,
        cbas_update_epochs=args.cbas_update_epochs,
        num_replicates=args.num_replicates,
        den_seed_pretrain_epochs=args.den_seed_pretrain_epochs,
    )
    summary = run_tfbind8_benchmark(
        checkpoint_root=args.checkpoint_root,
        output_dir=args.output_dir,
        raw_path=args.raw_path,
        device=args.device,
        config=config,
        resume=args.resume,
    )
    columns = [
        "weak_model_fraction",
        "algorithm",
        "best_true_score",
        "mean_true_score",
        "top_10_mean_true_score",
        "top_1pct_count",
        "mean_pairwise_hamming",
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
