"""Generic command-line runner for a user-provided Python scorer callable."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .algorithms import available_algorithms, run
from .api import CallableScorer, OptimizationConfig, SequenceSpace
from .objectives import TargetObjective
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", required=True, help="Python callable: package.module:function")
    parser.add_argument("--algorithm", choices=available_algorithms(), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--alphabet", default="ACGT")
    parser.add_argument("--budget", type=int, required=True)
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
    result = run(
        args.algorithm,
        scorer,
        space,
        OptimizationConfig(args.budget, num_candidates=args.num_candidates),
        initial=initial,
        objective=objective,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_frame().to_csv(output, index=False)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "algorithm": result.algorithm,
                "query_count": result.query_count,
                "initialization": initialization,
                "seed": args.seed,
                "task_names": list(result.task_names),
                "metadata": result.metadata,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
