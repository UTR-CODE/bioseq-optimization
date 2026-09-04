"""Feature handling for the Cetnar et al. (2024) LightGBM reproduction."""

from __future__ import annotations

import pandas as pd


NUM_TOP_ISOFORMS = 5
TIMEPOINTS = ("T0", "T2", "T4", "T8", "T16")


def paper_feature_columns(decay_model: bool = False) -> list[str]:
    """Return the exact feature family selected in ModelTrainingTesting.py."""
    columns: list[str] = []
    for isoform_index in range(NUM_TOP_ISOFORMS):
        suffix = f"_iso_{isoform_index}"
        features = [
            "rppH", "TX_rate", "TL_rate", "initial_sum_external_ssRNA", "initial_dsRNA_sum",
            "initial_external_ssRNA_A", "initial_external_ssRNA_T",
            "initial_external_ssRNA_G", "initial_external_ssRNA_C", "GQUAD", "iMOTIF", "CsrA",
            *[f"nt_{position}" for position in range(80)],
        ]
        columns.extend(feature + suffix for feature in features)
    if decay_model:
        columns.extend(f"predicted_{timepoint}" for timepoint in TIMEPOINTS)
    return columns


def categorise_paper_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the categorical dtype conversions from the authors' script."""
    frame = frame.copy()
    for isoform_index in range(NUM_TOP_ISOFORMS):
        suffix = f"_iso_{isoform_index}"
        for feature in ("rppH", "GQUAD", "iMOTIF", "CsrA", *[f"nt_{position}" for position in range(80)]):
            column = feature + suffix
            frame[column] = frame[column].astype("category")
    return frame


PAPER_LIGHTGBM_PARAMS = {
    "task": "train",
    "boosting_type": "gbdt",
    "linear_tree": False,
    "force_row_wise": False,
    "force_col_wise": True,
    "objective": "l2",
    "metric": "l2",
    "num_leaves": 100,
    "max_depth": 5,
    "max_bin": 1000,
    "n_estimators": 119,
    "learning_rate": 0.1,
    "bagging_fraction": 0.50,
    "bagging_freq": 5,
    "feature_fraction": 0.25,
    "min_data_in_leaf": 50,
    "importance_type": "gain",
}
