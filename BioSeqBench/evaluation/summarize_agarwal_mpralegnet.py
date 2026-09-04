"""Summarise the practical 10-fold MPRALegNet protocol for lentiMPRA.

The source-compatible benchmark protocol trains one outer held-out model per
publication fold.  Metrics are therefore reported as the mean and sample
standard deviation across its ten independently trained outer-fold models.
They are not a pooled Pearson correlation: per-sequence out-of-fold
predictions are intentionally not retained by the training runner.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CELL_DATASETS = {
    "hepg2": "agarwal2025_lentimpra_hepg2",
    "k562": "agarwal2025_lentimpra_k562",
    "wtc11": "agarwal2025_lentimpra_wtc11",
}
METRICS = ("pearson", "spearman", "mse", "r2")


def canonical_fold_pairs() -> list[tuple[int, int]]:
    """One fixed validation fold for each held-out publication fold."""
    return [(test_fold % 10 + 1, test_fold) for test_fold in range(1, 11)]


def read_results(checkpoint_dir: Path, dataset: str) -> list[dict]:
    results = []
    for validation_fold, test_fold in canonical_fold_pairs():
        path = checkpoint_dir / dataset / f"model_{validation_fold}_{test_fold}" / "result.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed canonical fold: {path}")
        value = json.loads(path.read_text())
        if value.get("smoke_limited"):
            raise ValueError(f"Smoke-test result cannot be summarised: {path}")
        results.append(value)
    return results


def summarise_cell(cell_type: str, results: list[dict]) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": CELL_DATASETS[cell_type],
        "model": "mpralegnet",
        "cell_type": cell_type,
        "outer_folds": len(results),
        "oof_test_rows": int(sum(result["test_rows"] for result in results)),
        "parameter_count": int(results[0]["parameter_count"]),
        "best_epoch_mean": float(np.mean([result["best_epoch"] for result in results])),
        "best_epoch_min": int(min(result["best_epoch"] for result in results)),
        "best_epoch_max": int(max(result["best_epoch"] for result in results)),
        "aggregation": "mean and sample SD over ten outer folds; test-time forward/RC mean prediction (TTA)",
    }
    weights = np.asarray([result["test_rows"] for result in results], dtype=float)
    for metric in METRICS:
        values = np.asarray([result[f"test_tta_{metric}"] for result in results], dtype=float)
        row[f"tta_{metric}_mean"] = float(values.mean())
        row[f"tta_{metric}_std"] = float(values.std(ddof=1))
        row[f"tta_{metric}_weighted_mean"] = float(np.average(values, weights=weights))
    return row


def evaluate(args: argparse.Namespace) -> list[dict[str, object]]:
    checkpoint_dir = Path(args.checkpoint_dir)
    rows = [summarise_cell(cell, read_results(checkpoint_dir, dataset)) for cell, dataset in CELL_DATASETS.items()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarise lentiMPRA MPRALegNet 10-fold results")
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints/strong_agarwal2025_mpralegnet_v1")
    parser.add_argument("--output", default="BioSeqBench/results/agarwal2025_mpralegnet_cv_summary.csv")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
