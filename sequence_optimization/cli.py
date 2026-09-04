"""Generic command-line runner for a user-provided Python scorer callable."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .algorithms import available_algorithms, run
from .api import CallableScorer, OptimizationConfig, SequenceSpace
from .objectives import TargetObjective
from .saturation import SaturationConfig, optimize_to_saturation
from .seeds import load_seed_sequences, random_seed_sequences


def _load_callable(specification: str):
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("--scorer must have the form package.module:function")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{specification!r} is not callable")
    return function


def _target(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _budget(value: str) -> int | None:
    if value.strip().lower() in {"unlimited", "unlimit", "none"}:
        return None
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("budget must be positive or 'unlimited'")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scorer", required=True, help="Python callable: package.module:function"
    )
    parser.add_argument("--algorithm", choices=available_algorithms(), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--alphabet", default="ACGT")
    parser.add_argument(
        "--budget",
        type=_budget,
        required=True,
        help="A positive query count, or 'unlimited' to stop at a plateau",
    )
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-file", help="Optional one-sequence-per-line seed file")
    parser.add_argument(
        "--task-names",
        help="Optional comma-separated model output names, e.g. developmental,housekeeping",
    )
    parser.add_argument(
        "--target",
        default="0",
        help="Output index or a name supplied in --task-names (default: 0)",
    )
    parser.add_argument(
        "--objective",
        choices=("maximize", "specificity"),
        default="maximize",
        help="Maximize the target alone or penalize all non-target outputs",
    )
    parser.add_argument(
        "--other-penalty",
        type=float,
        default=1.0,
        help="Non-target penalty multiplier for the specificity objective",
    )
    parser.add_argument(
        "--saturation-fractions",
        default="0.8,0.9",
        help="Selections to retain with --budget unlimited (default: 0.8,0.9)",
    )
    parser.add_argument("--minimum-queries", type=int)
    parser.add_argument("--plateau-patience-queries", type=int)
    parser.add_argument("--maximum-queries", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    space = SequenceSpace(args.length, args.alphabet)
    if args.seed_file:
        initial = load_seed_sequences(args.seed_file, space, count=args.num_candidates)
        initialization = "user_seed_file"
    else:
        initial = random_seed_sequences(space, args.num_candidates, seed=args.seed)
        initialization = "random"
    task_names = None
    if args.task_names:
        task_names = tuple(name.strip() for name in args.task_names.split(","))
    scorer = CallableScorer(_load_callable(args.scorer), task_names=task_names)
    objective = TargetObjective(
        target=_target(args.target),
        mode=args.objective,
        penalty=args.other_penalty,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.budget is None:
        import pandas as pd

        fractions = tuple(
            float(value.strip()) for value in args.saturation_fractions.split(",")
        )
        saturated = optimize_to_saturation(
            scorer,
            space,
            algorithm=args.algorithm,
            num_candidates=args.num_candidates,
            initial=initial,
            objective=objective,
            saturation=SaturationConfig(
                fractions=fractions,
                minimum_queries=args.minimum_queries,
                plateau_patience_queries=args.plateau_patience_queries,
                maximum_queries=args.maximum_queries,
            ),
            seed=args.seed,
        )
        frames = []
        for fraction, selection in saturated.selections.items():
            frame = selection.to_frame()
            frame.insert(0, "saturation_fraction", fraction)
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(output, index=False)
        run_metadata = {
            "mode": "saturation",
            "algorithm": args.algorithm,
            "total_query_count": saturated.total_query_count,
            "knee_query_count": saturated.knee_query_count,
            "stop_reason": saturated.stop_reason,
            "selection_query_counts": {
                str(fraction): result.query_count
                for fraction, result in saturated.selections.items()
            },
        }
        task_names = next(iter(saturated.selections.values())).task_names
    else:
        result = run(
            args.algorithm,
            scorer,
            space,
            OptimizationConfig(args.budget, num_candidates=args.num_candidates),
            initial=initial,
            objective=objective,
            seed=args.seed,
        )
        result.to_frame().to_csv(output, index=False)
        run_metadata = {
            "mode": "fixed_budget",
            "algorithm": result.algorithm,
            "query_count": result.query_count,
            "metadata": result.metadata,
        }
        task_names = result.task_names
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "initialization": initialization,
                "seed": args.seed,
                "task_names": list(task_names),
                **run_metadata,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
