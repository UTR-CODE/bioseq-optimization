"""Generic command-line runner for a user-provided Python scorer callable."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd

from .algorithms import available_algorithms, optimize
from .api import CallableScorer
from .objectives import TargetObjective
from .seeds import load_seed_sequences, random_seed_sequences


def _load_callable(specification: str, option: str):
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError(f"{option} must have the form package.module:function")
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
    parser.add_argument(
        "--scorer", required=True, help="Python callable: package.module:function"
    )
    parser.add_argument(
        "--constraint",
        action="append",
        help=(
            "Optional package.module:function returning pass probabilities; "
            "repeat for independent intersected constraints"
        ),
    )
    parser.add_argument("--algorithm", choices=available_algorithms(), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--alphabet", default="ACGT")
    parser.add_argument(
        "--pad-to-length",
        type=int,
        help="Optional fixed model input length; shorter sequences are padded with N",
    )
    parser.add_argument(
        "--padding-side",
        choices=("left", "right"),
        default="right",
        help="Side on which N padding is added (default: right)",
    )
    search = parser.add_mutually_exclusive_group()
    search.add_argument("--gain-fraction", type=float, action="append",
                        help="Plateau gain fraction to retain (repeatable, default: 0.9)")
    search.add_argument("--score-threshold", type=float, help="Stop at this scalar objective score")
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--diversity", type=int, default=5)
    parser.add_argument(
        "--fixed-region",
        action="append",
        help="Repeatable zero-based position or Python-style start:stop slice",
    )
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

    if args.seed_file:
        initial = load_seed_sequences(
            args.seed_file,
            args.length,
            alphabet=args.alphabet,
            count=args.num_candidates,
        )
        initialization = "user_seed_file"
    else:
        initial = random_seed_sequences(
            args.length, args.num_candidates, seed=args.seed, alphabet=args.alphabet
        )
        initialization = "random"
    task_names = None
    if args.task_names:
        task_names = tuple(name.strip() for name in args.task_names.split(","))
    scorer = CallableScorer(
        _load_callable(args.scorer, "--scorer"),
        task_names=task_names,
        pad_to_length=args.pad_to_length,
        padding_side=args.padding_side,
    )
    objective = TargetObjective(
        target=_target(args.target),
        mode=args.objective,
        penalty=args.other_penalty,
    )
    constraint = (
        [
            _load_callable(specification, "--constraint")
            for specification in args.constraint
        ]
        if args.constraint
        else None
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fractions = args.gain_fraction or []
    if any(not 0 < fraction <= 1 for fraction in fractions):
        parser.error("--gain-fraction must lie in (0, 1]")
    if len(set(fractions)) != len(fractions):
        parser.error("--gain-fraction values must be distinct")
    if args.score_threshold is not None:
        from math import isfinite
        if not isfinite(args.score_threshold):
            parser.error("--score-threshold must be finite")
    result = optimize(
        scorer,
        args.length,
        initial,
        objective,
        args.num_candidates,
        algorithm=args.algorithm,
        diversity=args.diversity,
        fixed_region=args.fixed_region,
        alphabet=args.alphabet,
        constraint=constraint,
        gain_fraction=fractions[0] if len(fractions) == 1 else None,
        gain_fractions=fractions if len(fractions) > 1 else None,
        score_threshold=args.score_threshold,
        seed=args.seed,
    )
    if len(fractions) > 1:
        stages = [result.at(fraction) for fraction in fractions]
        pd.concat(
            [stage.to_frame().assign(gain_fraction=fraction)
             for fraction, stage in zip(fractions, stages)],
            ignore_index=True,
        ).to_csv(output, index=False)
        run_metadata = {
            "mode": "gain",
            "algorithm": stages[0].algorithm,
            "query_count": result.total_query_count,
            "stop_reason": result.stop_reason,
            "gain_stages": [
                dict(gain_fraction=fraction, selection_query_count=stage.query_count,
                     metadata=stage.metadata)
                for fraction, stage in zip(fractions, stages)
            ],
        }
        task_names = stages[0].task_names
    else:
        result.to_frame().to_csv(output, index=False)
        run_metadata = {
            "mode": "gain" if args.score_threshold is None else "threshold",
            "algorithm": result.algorithm,
            "query_count": result.metadata["total_convergence_queries"],
            "selection_query_count": result.query_count,
            "stop_reason": result.metadata["stop_reason"],
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
