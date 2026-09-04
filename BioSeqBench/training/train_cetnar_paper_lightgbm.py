"""Reproduce Cetnar et al.'s feature-based mRNA Stability Calculator model."""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.cetnar2024_mrna_stability import (
    PAPER_LIGHTGBM_PARAMS,
    TIMEPOINTS,
    categorise_paper_features,
    paper_feature_columns,
)
from BioSeqBench.training.utils import set_seed


def read_test_table(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open("TestDataset.csv") as handle:
            return pd.read_csv(handle)


def outcomes_by_index(supplementary_path: Path) -> pd.DataFrame:
    # The original code uses ``iloc[Index - 1]``; make this one-based mapping
    # explicit so that the provenance is preserved even if pandas reindexes.
    outcomes = pd.read_excel(supplementary_path)
    outcomes.index = np.arange(1, len(outcomes) + 1)
    return outcomes


def timepoint_target(frame: pd.DataFrame, outcomes: pd.DataFrame, timepoint: str) -> np.ndarray:
    selected = outcomes.loc[frame["Index"].to_numpy()]
    ratio = selected[f"RNA_{timepoint}"].to_numpy(dtype=float) / selected["DNA"].to_numpy(dtype=float)
    result = np.full(len(frame), np.nan, dtype=float)
    valid = np.isfinite(ratio) & (ratio > 0)
    result[valid] = np.log(ratio[valid])
    return result


def fit_model(features_train: pd.DataFrame, target_train: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**PAPER_LIGHTGBM_PARAMS, verbosity=-1)
    model.fit(features_train, target_train)
    return model


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    raw_dir = Path(args.raw_root)
    train_frame = pd.read_csv(raw_dir / "TrainingDataset.csv")
    test_frame = read_test_table(raw_dir / "TestDataset.csv.zip")
    outcomes = outcomes_by_index(raw_dir / "SupplementaryData1.xlsx")
    train_frame = categorise_paper_features(train_frame)
    test_frame = categorise_paper_features(test_frame)
    base_columns = paper_feature_columns()
    # Preserve original dataframe order, as ``df.columns.isin(keepColumns)`` did.
    base_columns = [column for column in train_frame.columns if column in set(base_columns)]
    output_dir = Path(args.checkpoint_dir) / "cetnar2024_mrna_stability" / "paper_lightgbm"
    output_dir.mkdir(parents=True, exist_ok=True)
    timepoint_scores: dict[str, dict[str, float]] = {}
    for timepoint in TIMEPOINTS:
        target_train = timepoint_target(train_frame, outcomes, timepoint)
        target_test = timepoint_target(test_frame, outcomes, timepoint)
        valid_train = np.isfinite(target_train)
        valid_test = np.isfinite(target_test)
        model = fit_model(train_frame.loc[valid_train, base_columns], target_train[valid_train])
        train_frame[f"predicted_{timepoint}"] = model.predict(train_frame[base_columns])
        test_frame[f"predicted_{timepoint}"] = model.predict(test_frame[base_columns])
        model.booster_.save_model(str(output_dir / f"model_{timepoint}.txt"))
        timepoint_scores[timepoint] = {
            "train_r2": float(r2_score(target_train[valid_train], train_frame.loc[valid_train, f"predicted_{timepoint}"])),
            "test_r2": float(r2_score(target_test[valid_test], test_frame.loc[valid_test, f"predicted_{timepoint}"])),
            "train_rows": int(valid_train.sum()), "test_rows": int(valid_test.sum()),
        }
    decay_columns = paper_feature_columns(decay_model=True)
    decay_columns = [column for column in train_frame.columns if column in set(decay_columns)]
    target_train = outcomes.loc[train_frame["Index"].to_numpy(), "k_fit"].to_numpy(dtype=float)
    target_test = outcomes.loc[test_frame["Index"].to_numpy(), "k_fit"].to_numpy(dtype=float)
    decay_model = fit_model(train_frame[decay_columns], target_train)
    train_prediction = decay_model.predict(train_frame[decay_columns])
    test_prediction = decay_model.predict(test_frame[decay_columns])
    decay_model.booster_.save_model(str(output_dir / "decay_rate.txt"))
    full_metrics = compute_metrics(target_test, test_prediction)
    positive = target_test > 0
    physical_metrics = compute_metrics(target_test[positive], test_prediction[positive])
    train_sequences = set(train_frame["UTR_Designed_Sequence"].str.replace("T", "U"))
    unseen = ~test_frame["UTR_Designed_Sequence"].str.replace("T", "U").isin(train_sequences).to_numpy()
    unseen_positive = unseen & positive
    unseen_metrics = compute_metrics(target_test[unseen_positive], test_prediction[unseen_positive])
    predictions = pd.DataFrame({
        "Index": test_frame["Index"], "sequence": test_frame["UTR_Designed_Sequence"].str.replace("T", "U"),
        "true_k_fit": target_test, "predicted_k_fit": test_prediction,
    })
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    result: dict[str, object] = {
        "timestamp": int(time.time()), "dataset": "cetnar2024_mrna_stability", "model": "paper_lightgbm",
        "lightgbm_version": lgb.__version__, "seed": args.seed,
        "source_train_rows": len(train_frame), "source_test_rows": len(test_frame),
        "feature_count_timepoint": len(base_columns), "feature_count_decay": len(decay_columns),
        "paper_decay_test_r2": 0.43, "paper_timepoint_test_r2": {"T0": 0.69, "T2": 0.65, "T4": 0.60, "T8": 0.53, "T16": 0.43},
        "timepoint_models": timepoint_scores,
        "full_source_test": full_metrics,
        "positive_k_fit_benchmark_test": physical_metrics,
        "positive_k_fit_test_rows": int(positive.sum()),
        "cross_split_repeated_sequence_test_rows": int((~unseen).sum()),
        "strict_unseen_sequence_test": unseen_metrics,
        "strict_unseen_sequence_test_rows": int(unseen_positive.sum()),
    }
    write_result(output_dir / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cetnar 2024 paper LightGBM reproduction")
    parser.add_argument("--raw-root", default="BioSeqBench/raw_data/cetnar2024_mrna_stability")
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints/strong_cetnar_paper_v1")
    parser.add_argument("--seed", type=int, default=1, help="Matches the authors' Python random seed")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
