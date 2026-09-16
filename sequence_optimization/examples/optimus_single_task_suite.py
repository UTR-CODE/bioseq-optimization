"""Optimize Optimus 5-Prime MRL with a selectable optimizer suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    )
    parser.add_argument("--initial-count", type=int, default=1_000)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--model-batch-size", type=int, default=4_096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights-dir")
    parser.add_argument("--output-dir", default="optimus_mrl_suite")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model = load_model(
        "optimus5prime",
        device=args.device,
        batch_size=args.model_batch_size,
        weights_dir=args.weights_dir,
    )
    selected = tuple(args.algorithms)
    differentiable_model = None
    if set(selected) & DIFFERENTIABLE_ALGORITHMS:
        differentiable_model = SequencePredictor(
            model.model,
            device=model.device,
            task_names=model.task_names,
        )
    initial = random_seed_sequences(
        length=50,
        count=args.initial_count,
        seed=args.seed,
    )
    suite = optimize_algorithms(
        model,
        algorithms=selected,
        differentiable_model=differentiable_model,
        length=50,
        initial=initial,
        objective=["maximize"],
        num_candidates=args.num_candidates,
        budget=args.budget,
        seed=args.seed,
    )
    suite.save_csv(args.output_dir)
    Path(args.output_dir, "run_config.json").write_text(
        json.dumps(
            {
                "model": "optimus5prime",
                "objective": ["maximize"],
                "algorithms": list(selected),
                "initial_count": args.initial_count,
                "num_candidates": args.num_candidates,
                "budget": args.budget,
                "model_batch_size": args.model_batch_size,
                "device": args.device,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n"
    )
    print(suite.summary_frame().to_string(index=False))


if __name__ == "__main__":
    main()
