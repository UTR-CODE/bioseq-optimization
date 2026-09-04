"""CLI for the Optimus 5'UTR lower-tail optimization benchmark."""

from __future__ import annotations

import argparse

from BioSeqBench.optimization.utr5_benchmark import (
    ALGORITHM_NAMES,
    DEFAULT_FRACTIONS,
    UTR5OptimizationConfig,
    run_utr5_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_NAMES, default=list(ALGORITHM_NAMES))
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--query-budget", type=int, default=10_000)
    parser.add_argument("--num-replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--checkpoint-root", default="BioSeqBench/results/utr5_weak_cnns")
    parser.add_argument(
        "--strong-checkpoint",
        default="BioSeqBench/checkpoints/strong_test_guided_v1/utr5_mpra/optimus5prime/best.pt",
    )
    parser.add_argument("--data-root", default="BioSeqBench/dataset/utr5_mpra")
    parser.add_argument("--output-dir", default="BioSeqBench/results/utr5_optimization_all_models_5seeds_shared_init")
    parser.add_argument("--device")
    parser.add_argument("--cbas-rounds", type=int, default=10)
    parser.add_argument("--cbas-pretrain-epochs", type=int, default=20)
    parser.add_argument("--cbas-update-epochs", type=int, default=10)
    parser.add_argument("--den-seed-pretrain-epochs", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = UTR5OptimizationConfig(
        train_fractions=tuple(args.fractions),
        algorithms=tuple(args.algorithms),
        num_candidates=args.num_candidates,
        query_budget=args.query_budget,
        num_replicates=args.num_replicates,
        base_seed=args.seed,
        top_visible_initializations=args.num_candidates // 2,
        random_initializations=args.num_candidates - args.num_candidates // 2,
        cbas_rounds=args.cbas_rounds,
        cbas_pretrain_epochs=args.cbas_pretrain_epochs,
        cbas_update_epochs=args.cbas_update_epochs,
        den_seed_pretrain_epochs=args.den_seed_pretrain_epochs,
    )
    summary = run_utr5_benchmark(
        checkpoint_root=args.checkpoint_root,
        strong_checkpoint=args.strong_checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        config=config,
        resume=args.resume,
    )
    print(
        summary[
            ["weak_model_fraction", "algorithm", "mean_strong_score", "best_strong_score", "top_1pct_count"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
