"""Oracle-oriented diagnostics for the eight DREAM 2022 test subsets.

The DREAM test labels are on the public MAUDE scale while this model emits its
FACS-bin-derived score.  This intentional scale mismatch makes raw MSE/R² hard
to interpret, so this report emphasizes ranking, calibration, and the model's
own predicted head as required by sequence-optimization benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SUBSETS = (
    "all_random_seqs",
    "motif_perturbation",
    "high_exp_seqs",
    "all_SNVs_seqs",
    "low_exp_seqs",
    "yeast_seqs",
    "challenging_seqs",
    "motif_tiling_seqs",
)


def safe_spearman(prediction: np.ndarray, target: np.ndarray) -> float | None:
    value = float(spearmanr(prediction, target).statistic)
    return value if math.isfinite(value) else None


def subset_indices(archive: tarfile.TarFile, subset: str) -> tuple[np.ndarray, int]:
    """Read unique public-test row positions for a subset.

    Variant-pair files have both alt/ref row positions.  A test sequence is
    intentionally evaluated once even when it occurs in more than one pair.
    """
    with archive.extractfile(f"test_subset_ids/{subset}.csv") as handle:
        if handle is None:
            raise FileNotFoundError(subset)
        frame = pd.read_csv(handle)
    position_columns = [column for column in frame.columns if column.endswith("_pos")]
    if not position_columns:
        position_columns = ["pos"]
    positions = np.concatenate([frame[column].dropna().astype(np.int64).to_numpy() for column in position_columns])
    return np.unique(positions), len(frame)


def binned_spearman(prediction: np.ndarray, target: np.ndarray, bin_size: int) -> tuple[list[dict[str, float | int | None]], dict | None]:
    order = np.argsort(prediction, kind="mergesort")
    prediction, target = prediction[order], target[order]
    bins: list[dict[str, float | int | None]] = []
    ceiling: dict | None = None
    for start in range(0, len(prediction), bin_size):
        stop = min(start + bin_size, len(prediction))
        score = safe_spearman(prediction[start:stop], target[start:stop])
        entry: dict[str, float | int | None] = {
            "bin": start // bin_size + 1,
            "n": stop - start,
            "prediction_min": float(prediction[start]),
            "prediction_max": float(prediction[stop - 1]),
            "prediction_mean": float(prediction[start:stop].mean()),
            "spearman": score,
        }
        bins.append(entry)
        if ceiling is None and score is not None and score <= 0:
            ceiling = entry
    return bins, ceiling


def diagnose(prediction: np.ndarray, target: np.ndarray, *, bin_size: int, top_k: int) -> dict:
    if len(prediction) != len(target) or len(prediction) < 2:
        raise ValueError("Prediction/target arrays must have the same length >= 2")
    # true = slope * prediction + intercept.  Lstsq avoids silently fitting a
    # different convention from the specification in the benchmark README.
    design = np.column_stack([prediction, np.ones(len(prediction))])
    slope, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
    head_n = max(1, math.ceil(0.10 * len(prediction)))
    order_prediction = np.argsort(-prediction, kind="mergesort")
    order_target = np.argsort(-target, kind="mergesort")
    head = order_prediction[:head_n]
    k = min(top_k, len(prediction))
    hits = len(np.intersect1d(order_prediction[:k], order_target[:k], assume_unique=True))
    bins, ceiling = binned_spearman(prediction, target, bin_size)
    return {
        "spearman": safe_spearman(prediction, target),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "head_fraction": 0.10,
        "head_n": head_n,
        "head_bias_true_minus_pred": float(target[head].mean() - prediction[head].mean()),
        "top_k": k,
        "top_k_hit_count": hits,
        "top_k_hit_rate": hits / k,
        "bin_size": bin_size,
        "binned_spearman": bins,
        "first_nonpositive_spearman_bin": ceiling,
    }


def csv_row(category: str, records: int, rows: int, result: dict) -> dict[str, object]:
    return {
        "category": category,
        "source_records": records,
        "test_sequences": rows,
        "spearman": result["spearman"],
        "calibration_slope": result["calibration_slope"],
        "head_bias_true_minus_pred": result["head_bias_true_minus_pred"],
        "top_100_hit_rate": result["top_k_hit_rate"],
        "top_100_hit_count": result["top_k_hit_count"],
        "binned_spearman_json": json.dumps(result["binned_spearman"], separators=(",", ":")),
    }


def evaluate(args: argparse.Namespace) -> dict:
    arrays = np.load(args.predictions)
    target = np.asarray(arrays["y_true"], dtype=np.float64)
    prediction = np.asarray(arrays[args.prediction_key], dtype=np.float64)
    if len(target) != len(prediction):
        raise ValueError("Prediction array length does not match y_true")
    results: dict[str, object] = {
        "dataset": "yeast_gpra_dream2022",
        "prediction_key": args.prediction_key,
        "test_rows": len(target),
        "subset_policy": "Unique test rows per subset; paired alt/ref records contribute both positions, then duplicate positions are removed.",
        "metrics": {},
    }
    rows: list[dict[str, object]] = []
    with tarfile.open(args.subsets, "r:gz") as archive:
        for subset in SUBSETS:
            indices, records = subset_indices(archive, subset)
            if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(target):
                raise ValueError(f"Invalid test indices in {subset}")
            score = diagnose(prediction[indices], target[indices], bin_size=args.bin_size, top_k=args.top_k)
            results["metrics"][subset] = {"source_records": records, "test_sequences": len(indices), **score}
            rows.append(csv_row(subset, records, len(indices), score))
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_suffix(".json")).write_text(json.dumps(results, indent=2) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(results, indent=2), flush=True)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DREAM 2022 subset oracle diagnostics")
    parser.add_argument("--predictions", default="BioSeqBench/checkpoints/strong_yeast_dream2022_autosome_v3_validation_5k/yeast_gpra_dream2022/autosome_legnet/test_predictions.npz")
    parser.add_argument("--prediction-key", default="best_val_prediction", choices=("best_val_prediction", "last_prediction"))
    parser.add_argument("--subsets", default="BioSeqBench/raw_data/yeast_gpra_dream2022/test_subset_ids.tar.gz")
    parser.add_argument("--bin-size", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output-prefix", default="BioSeqBench/results/dream2022_oracle_diagnostics")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
