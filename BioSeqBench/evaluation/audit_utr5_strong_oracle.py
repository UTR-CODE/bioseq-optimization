"""Audit Optimus strong-oracle tail behavior and optimizer extrapolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from BioSeqBench.evaluation.metrics import compute_metrics


def run(result_dir: Path, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_root / "train.csv", usecols=["sequence", "label"])
    test = pd.read_csv(data_root / "test.csv", usecols=["sequence", "label"])
    reference = pd.concat(
        [train.assign(split="train"), test.assign(split="test")], ignore_index=True
    )
    with np.load(result_dir / "strong_reference_predictions.npz", allow_pickle=False) as archive:
        reference["strong_score"] = np.asarray(archive["strong_scores"], dtype=np.float64)
    diagnostic_rows = []
    scopes = {
        "train_all": reference["split"].eq("train"),
        "test_all": reference["split"].eq("test"),
        "combined_all": np.ones(len(reference), dtype=bool),
    }
    for tail in (0.05, 0.01, 0.001):
        count = max(2, int(round(len(reference) * tail)))
        selected = reference.nlargest(count, "label").index
        scopes[f"combined_measured_top_{tail * 100:g}pct"] = reference.index.isin(selected)
    for scope, mask in scopes.items():
        group = reference.loc[mask]
        metrics = compute_metrics(group["label"].to_numpy(), group["strong_score"].to_numpy())
        diagnostic_rows.append(
            {
                "scope": scope,
                "count": len(group),
                "measured_min": group["label"].min(),
                "measured_max": group["label"].max(),
                "strong_min": group["strong_score"].min(),
                "strong_max": group["strong_score"].max(),
                **metrics,
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(result_dir / "strong_oracle_tail_diagnostics.csv", index=False)

    candidates = pd.read_csv(result_dir / "all_candidates.csv")
    reference_max = float(reference["strong_score"].max())
    extrapolation = candidates.assign(
        above_reference_max=candidates["strong_score"] > reference_max,
        margin_above_reference_max=candidates["strong_score"] - reference_max,
    ).groupby(["weak_model_fraction", "algorithm"], sort=False).agg(
        candidate_count=("sequence", "size"),
        above_reference_max_count=("above_reference_max", "sum"),
        above_reference_max_fraction=("above_reference_max", "mean"),
        maximum_strong_score=("strong_score", "max"),
        maximum_margin_above_reference=("margin_above_reference_max", "max"),
    ).reset_index()
    extrapolation.to_csv(result_dir / "strong_oracle_extrapolation.csv", index=False)
    conservative_runs = candidates.assign(
        clipped_strong_score=candidates["strong_score"].clip(upper=reference_max),
        within_reference_range=candidates["strong_score"] <= reference_max,
    ).groupby(["weak_model_fraction", "replicate", "algorithm"], sort=False).agg(
        mean_clipped_strong_score=("clipped_strong_score", "mean"),
        best_within_reference_score=(
            "strong_score",
            lambda values: values[values <= reference_max].max()
            if (values <= reference_max).any()
            else np.nan,
        ),
        above_reference_max_count=("within_reference_range", lambda values: int((~values).sum())),
    ).reset_index()
    ranked = conservative_runs[conservative_runs["algorithm"] != "training_top100"].copy()
    ranked["rank"] = ranked.groupby(["weak_model_fraction", "replicate"])[
        "mean_clipped_strong_score"
    ].rank(ascending=False, method="average")
    conservative = ranked.groupby("algorithm", sort=False).agg(
        conservative_mean_rank=("rank", "mean"),
        mean_clipped_strong_score=("mean_clipped_strong_score", "mean"),
        mean_best_within_reference_score=("best_within_reference_score", "mean"),
        above_reference_max_count=("above_reference_max_count", "sum"),
        conditions=("rank", "size"),
    ).reset_index().sort_values(["conservative_mean_rank", "algorithm"])
    conservative.to_csv(result_dir / "strong_oracle_conservative_ranking.csv", index=False)
    audit = {
        "strong_reference_count": len(reference),
        "strong_reference_max": reference_max,
        "generated_candidate_count": len(candidates),
        "generated_above_reference_max_count": int((candidates["strong_score"] > reference_max).sum()),
        "generated_above_reference_max_fraction": float((candidates["strong_score"] > reference_max).mean()),
        "generated_max": float(candidates["strong_score"].max()),
        "interpretation": (
            "Scores above the maximum strong prediction on the canonical 280k library are surrogate "
            "extrapolation signals, not experimentally validated MRL improvements."
        ),
    }
    (result_dir / "strong_oracle_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return diagnostics, extrapolation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="BioSeqBench/results/utr5_optimization_all_models_5seeds_shared_init",
    )
    parser.add_argument("--data-root", default="BioSeqBench/dataset/utr5_mpra")
    args = parser.parse_args()
    diagnostics, extrapolation = run(Path(args.result_dir), Path(args.data_root))
    print(diagnostics.to_string(index=False))
    print(extrapolation.groupby("algorithm").agg(
        above_reference_max_count=("above_reference_max_count", "sum"),
        candidate_count=("candidate_count", "sum"),
        maximum_strong_score=("maximum_strong_score", "max"),
    ).sort_values("above_reference_max_count", ascending=False).to_string())


if __name__ == "__main__":
    main()
