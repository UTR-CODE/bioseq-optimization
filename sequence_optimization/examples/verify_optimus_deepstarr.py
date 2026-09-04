"""Reproduce the Optimus and DeepSTARR optimization smoke tests."""

from __future__ import annotations

import json

import numpy as np

from sequence_optimization import (
    ISM_optimize,
    SequenceSpace,
    TargetObjective,
    genetic_algorithm_optimize,
)
from sequence_optimization.examples.models import DeepSTARRExample, Optimus5PrimeExample


def main() -> None:
    optimus_seeds = np.asarray([("ACGT" * 13)[:50], ("TGCA" * 13)[:50]])
    optimus = Optimus5PrimeExample.from_checkpoint(device="cpu", batch_size=512)
    optimus_result = ISM_optimize(
        optimus,
        SequenceSpace(50, "ACGT"),
        budget=302,
        num_candidates=2,
        initial=optimus_seeds,
        seed=11,
    )

    deepstarr_seeds = np.asarray([
        "A" * 249,
        "C" * 249,
        "G" * 249,
        "T" * 249,
    ])
    deepstarr = DeepSTARRExample.from_checkpoint(device="cpu", batch_size=32)
    deepstarr_result = genetic_algorithm_optimize(
        deepstarr,
        SequenceSpace(249, "ACGT"),
        budget=8,
        num_candidates=4,
        initial=deepstarr_seeds,
        objective=TargetObjective(target="developmental", mode="specificity"),
        seed=11,
    )

    if optimus_result.query_count != 302 or optimus_result.task_names != ("mrl",):
        raise RuntimeError("Optimus smoke test did not satisfy its result contract")
    if deepstarr_result.query_count != 8 or deepstarr_result.task_names != (
        "developmental",
        "housekeeping",
    ):
        raise RuntimeError("DeepSTARR smoke test did not satisfy its result contract")

    report = {
        "optimus": {
            "algorithm": optimus_result.algorithm,
            "query_count": optimus_result.query_count,
            "best_sequence": str(optimus_result.sequences[0]),
            "best_mrl": float(optimus_result.activities[0, 0]),
        },
        "deepstarr": {
            "algorithm": deepstarr_result.algorithm,
            "query_count": deepstarr_result.query_count,
            "objective": deepstarr_result.metadata["objective"],
            "best_sequence": str(deepstarr_result.sequences[0]),
            "best_utility": float(deepstarr_result.scores[0]),
            "best_developmental": float(deepstarr_result.activities[0, 0]),
            "best_housekeeping": float(deepstarr_result.activities[0, 1]),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
