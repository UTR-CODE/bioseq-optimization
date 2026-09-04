"""Analyze Train-Top versus full-space-random seeds using existing TFBind8 runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BioSeqBench.oracles.tfbind import codes_to_encoded, load_tfbind_landscape


FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SEEDED_ALGORITHMS = (
    "ism_greedy",
    "simulated_annealing",
    "genetic_algorithm",
    "adalead",
    "vanilla_activation_maximization",
    "fast_seqprop",
    "cbas",
    "den",
)
ALGORITHM_LABELS = ("ISM", "SA", "GA", "AdaLead", "Vanilla AM", "Fast SeqProp", "CbAS", "DEN")
SOURCE_LABELS = {"train_top": "Train Top-50", "full_space_random": "Full-space Random-50"}
SOURCE_COLORS = {"train_top": "#2166ac", "full_space_random": "#d6604d"}


def _save(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def build_seed_statistics(
    result_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = pd.read_csv(result_dir / "all_shared_initial_sequences.csv")
    landscape = load_tfbind_landscape("tfbind8")
    ranked = landscape.ranked_codes()
    rank_by_code = np.empty(landscape.size, dtype=np.int64)
    rank_by_code[ranked] = np.arange(landscape.size)
    codes = seeds["sequence_code"].to_numpy(dtype=np.int64)
    seeds["true_score"] = landscape.score_codes(codes)
    seeds["activity_percentile"] = 100.0 * (rank_by_code[codes] + 1) / landscape.size
    seeds["weak_prediction"] = np.nan

    for fraction in FRACTIONS:
        mask = np.isclose(seeds["weak_model_fraction"], fraction)
        model_name = f"bottom_{fraction * 100:g}pct"
        with np.load(result_dir / model_name / "all_landscape_weak_predictions.npz") as archive:
            prediction = np.asarray(archive["mean"])
        seeds.loc[mask, "weak_prediction"] = prediction[
            seeds.loc[mask, "sequence_code"].to_numpy(dtype=np.int64)
        ]

    replicate_rows = []
    for (fraction, replicate, source), group in seeds.groupby(
        ["weak_model_fraction", "replicate", "source"], sort=False
    ):
        weak_best = group.loc[group["weak_prediction"].idxmax()]
        replicate_rows.append(
            {
                "weak_model_fraction": fraction,
                "replicate": replicate,
                "source": source,
                "true_mean": group["true_score"].mean(),
                "true_median": group["true_score"].median(),
                "true_best": group["true_score"].max(),
                "percentile_mean": group["activity_percentile"].mean(),
                "weak_mean": group["weak_prediction"].mean(),
                "weak_best": group["weak_prediction"].max(),
                "true_at_weak_best": weak_best["true_score"],
                "outside_training_fraction": 1.0 - group["in_weak_train"].mean(),
            }
        )
    replicate_frame = pd.DataFrame(replicate_rows)
    metrics = [
        "true_mean",
        "true_median",
        "true_best",
        "percentile_mean",
        "weak_mean",
        "weak_best",
        "true_at_weak_best",
        "outside_training_fraction",
    ]
    aggregations: dict[str, tuple[str, str]] = {}
    for metric in metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
        aggregations[f"{metric}_sem"] = (metric, "sem")
    summary = (
        replicate_frame.groupby(["weak_model_fraction", "source"], sort=False)
        .agg(**aggregations)
        .reset_index()
    )
    seeds.to_csv(result_dir / "initialization_sequences_scored.csv", index=False)
    replicate_frame.to_csv(result_dir / "initialization_source_replicates.csv", index=False)
    summary.to_csv(result_dir / "initialization_source_summary.csv", index=False)
    return replicate_frame, summary


def build_proximity_statistics(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = pd.read_csv(result_dir / "all_shared_initial_sequences.csv")
    candidates = pd.read_csv(result_dir / "all_candidates.csv")
    candidates = candidates[candidates["algorithm"].isin(SEEDED_ALGORITHMS)].copy()
    candidate_rows = []

    for (fraction, replicate, algorithm), group in candidates.groupby(
        ["weak_model_fraction", "replicate", "algorithm"], sort=False
    ):
        condition_seeds = seeds[
            np.isclose(seeds["weak_model_fraction"], fraction)
            & (seeds["replicate"] == replicate)
        ]
        candidate_encoded = codes_to_encoded(group["sequence_code"].to_numpy(), 8)
        distances = {}
        for source in ("train_top", "full_space_random"):
            seed_encoded = codes_to_encoded(
                condition_seeds.loc[
                    condition_seeds["source"] == source, "sequence_code"
                ].to_numpy(),
                8,
            )
            distances[source] = (
                candidate_encoded[:, None, :] != seed_encoded[None, :, :]
            ).sum(axis=2).min(axis=1)
        top_distance = distances["train_top"]
        random_distance = distances["full_space_random"]
        closer = np.where(
            top_distance < random_distance,
            "train_top",
            np.where(random_distance < top_distance, "full_space_random", "tie"),
        )
        for row, distance_top, distance_random, source in zip(
            group.to_dict("records"), top_distance, random_distance, closer
        ):
            candidate_rows.append(
                {
                    "weak_model_fraction": fraction,
                    "replicate": replicate,
                    "algorithm": algorithm,
                    "sequence": row["sequence"],
                    "sequence_code": row["sequence_code"],
                    "true_score": row["true_score"],
                    "minimum_train_top_distance": int(distance_top),
                    "minimum_full_random_distance": int(distance_random),
                    "closer_source": source,
                }
            )
    proximity = pd.DataFrame(candidate_rows)
    proximity.to_csv(result_dir / "final_candidate_seed_proximity.csv", index=False)
    summary = (
        proximity.assign(
            closer_train_top=proximity["closer_source"].eq("train_top"),
            closer_full_random=proximity["closer_source"].eq("full_space_random"),
            distance_tie=proximity["closer_source"].eq("tie"),
        )
        .groupby(["weak_model_fraction", "algorithm"], sort=False)
        .agg(
            closer_train_top_fraction=("closer_train_top", "mean"),
            closer_full_random_fraction=("closer_full_random", "mean"),
            distance_tie_fraction=("distance_tie", "mean"),
            mean_train_top_distance=("minimum_train_top_distance", "mean"),
            mean_full_random_distance=("minimum_full_random_distance", "mean"),
        )
        .reset_index()
    )
    summary["random_minus_train_proximity"] = (
        summary["closer_full_random_fraction"] - summary["closer_train_top_fraction"]
    )
    summary.to_csv(result_dir / "final_candidate_seed_proximity_summary.csv", index=False)
    return proximity, summary


def build_random_quality_correlations(
    result_dir: Path, replicate_seeds: pd.DataFrame
) -> pd.DataFrame:
    summary = pd.read_csv(result_dir / "summary.csv")
    final = summary[summary["algorithm"].isin(SEEDED_ALGORITHMS)].copy()
    random_stats = replicate_seeds[
        replicate_seeds["source"] == "full_space_random"
    ][["weak_model_fraction", "replicate", "true_mean", "true_best"]]
    merged = final.merge(
        random_stats,
        on=["weak_model_fraction", "replicate"],
        validate="many_to_one",
    )
    merged["random_mean_within_model"] = merged["true_mean"] - merged.groupby(
        "weak_model_fraction"
    )["true_mean"].transform("mean")
    merged["random_best_within_model"] = merged["true_best"] - merged.groupby(
        "weak_model_fraction"
    )["true_best"].transform("mean")
    merged["final_mean_within_model"] = merged["mean_true_score"] - merged.groupby(
        ["algorithm", "weak_model_fraction"]
    )["mean_true_score"].transform("mean")
    rows = []
    for algorithm, group in merged.groupby("algorithm", sort=False):
        rows.append(
            {
                "algorithm": algorithm,
                "random_seed_mean_vs_final_mean_correlation": group[
                    "random_mean_within_model"
                ].corr(group["final_mean_within_model"]),
                "random_seed_best_vs_final_mean_correlation": group[
                    "random_best_within_model"
                ].corr(group["final_mean_within_model"]),
                "conditions": len(group),
            }
        )
    correlations = pd.DataFrame(rows)
    correlations.to_csv(result_dir / "random_seed_quality_correlations.csv", index=False)
    return correlations


def build_decision_table(
    result_dir: Path,
    seed_summary: pd.DataFrame,
    proximity_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Write a compact, metric-specific comparison without claiming causality."""
    seed_wide = seed_summary.pivot(index="weak_model_fraction", columns="source")
    proximity = proximity_summary.groupby("weak_model_fraction").agg(
        closer_train_top_fraction=("closer_train_top_fraction", "mean"),
        closer_full_random_fraction=("closer_full_random_fraction", "mean"),
        distance_tie_fraction=("distance_tie_fraction", "mean"),
    )
    rows = []
    for fraction in FRACTIONS:
        random_mean_delta = (
            seed_wide.loc[fraction, ("true_mean_mean", "full_space_random")]
            - seed_wide.loc[fraction, ("true_mean_mean", "train_top")]
        )
        random_best_delta = (
            seed_wide.loc[fraction, ("true_best_mean", "full_space_random")]
            - seed_wide.loc[fraction, ("true_best_mean", "train_top")]
        )
        random_selected_delta = (
            seed_wide.loc[
                fraction, ("true_at_weak_best_mean", "full_space_random")
            ]
            - seed_wide.loc[fraction, ("true_at_weak_best_mean", "train_top")]
        )
        random_proximity_delta = (
            proximity.loc[fraction, "closer_full_random_fraction"]
            - proximity.loc[fraction, "closer_train_top_fraction"]
        )
        rows.append(
            {
                "weak_model_fraction": fraction,
                "higher_seed_mean_activity": (
                    "full_space_random" if random_mean_delta > 0 else "train_top"
                ),
                "random_minus_train_seed_mean_activity": random_mean_delta,
                "higher_best_seed_activity": (
                    "full_space_random" if random_best_delta > 0 else "train_top"
                ),
                "random_minus_train_best_seed_activity": random_best_delta,
                "higher_weak_selected_seed_activity": (
                    "full_space_random" if random_selected_delta > 0 else "train_top"
                ),
                "random_minus_train_weak_selected_activity": random_selected_delta,
                "final_candidate_proximity_preference": (
                    "full_space_random" if random_proximity_delta > 0 else "train_top"
                ),
                "random_minus_train_final_proximity": random_proximity_delta,
                "final_candidate_distance_tie_fraction": proximity.loc[
                    fraction, "distance_tie_fraction"
                ],
            }
        )
    decision = pd.DataFrame(rows)
    decision.to_csv(result_dir / "initialization_source_decision.csv", index=False)
    return decision


def plot_seed_quality(summary: pd.DataFrame, output_stem: Path) -> None:
    panels = (
        ("true_mean", "Mean true activity of 50 seeds", "True activity"),
        ("true_best", "Best true activity among 50 seeds", "True activity"),
        (
            "true_at_weak_best",
            "True activity of the seed ranked best by the weak model",
            "True activity",
        ),
        (
            "outside_training_fraction",
            "Fraction of seeds outside the weak-model training pool",
            "Fraction outside training",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.5), sharex=True, constrained_layout=True)
    x = np.asarray(FRACTIONS) * 100
    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        for source in ("train_top", "full_space_random"):
            rows = summary[summary["source"] == source].set_index(
                "weak_model_fraction"
            ).loc[list(FRACTIONS)]
            mean = rows[f"{metric}_mean"].to_numpy()
            interval = 1.96 * rows[f"{metric}_sem"].fillna(0).to_numpy()
            axis.plot(
                x,
                mean,
                marker="o" if source == "full_space_random" else "s",
                linewidth=2.1,
                color=SOURCE_COLORS[source],
                label=SOURCE_LABELS[source],
            )
            axis.fill_between(
                x,
                mean - interval,
                mean + interval,
                color=SOURCE_COLORS[source],
                alpha=0.14,
                linewidth=0,
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(x)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Weak-model lower-tail training fraction (%)")
    axes[1, 1].set_xlabel("Weak-model lower-tail training fraction (%)")
    figure.suptitle("TFBind8 initialization-source quality from existing shared-seed runs", fontsize=16)
    _save(figure, output_stem)


def plot_proximity_and_correlation(
    proximity: pd.DataFrame,
    correlations: pd.DataFrame,
    output_stem: Path,
) -> None:
    matrix = np.empty((len(FRACTIONS), len(SEEDED_ALGORITHMS)))
    for row, fraction in enumerate(FRACTIONS):
        for column, algorithm in enumerate(SEEDED_ALGORITHMS):
            selected = proximity[
                np.isclose(proximity["weak_model_fraction"], fraction)
                & (proximity["algorithm"] == algorithm)
            ]
            matrix[row, column] = selected.iloc[0]["random_minus_train_proximity"] * 100
    limit = max(float(np.abs(matrix).max()), 1.0)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(17.0, 7.5),
        gridspec_kw={"width_ratios": [2.2, 1]},
        constrained_layout=True,
    )
    image = axes[0].imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axes[0].text(
                column,
                row,
                f"{value:+.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.65 * limit else "#222222",
            )
    axes[0].set_title(
        "Final-candidate proximity: Random-closer minus Train-Top-closer (percentage points)",
        loc="left",
        fontweight="bold",
    )
    axes[0].set_xticks(range(len(ALGORITHM_LABELS)), ALGORITHM_LABELS, rotation=30, ha="right")
    axes[0].set_yticks(range(len(FRACTIONS)), [f"{x:g}%" for x in np.asarray(FRACTIONS) * 100])
    axes[0].set_ylabel("Weak-model training pool")
    figure.colorbar(image, ax=axes[0], fraction=0.03, pad=0.02)

    ordered = correlations.set_index("algorithm").loc[list(SEEDED_ALGORITHMS)]
    values = ordered["random_seed_mean_vs_final_mean_correlation"].to_numpy()
    colors = np.where(values >= 0, "#d6604d", "#4393c3")
    axes[1].barh(np.arange(len(values)), values, color=colors)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_yticks(range(len(ALGORITHM_LABELS)), ALGORITHM_LABELS)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1, 1)
    axes[1].set_xlabel("Within-model correlation")
    axes[1].set_title(
        "Random-seed mean activity vs final mean activity",
        loc="left",
        fontweight="bold",
    )
    axes[1].grid(axis="x", color="#dddddd", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    figure.suptitle("Evidence of initialization-source influence (not a causal ablation)", fontsize=16)
    _save(figure, output_stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="BioSeqBench/results/tfbind8_optimization_all_models_5seeds_shared_init",
    )
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    replicate_seeds, seed_summary = build_seed_statistics(result_dir)
    _, proximity_summary = build_proximity_statistics(result_dir)
    correlations = build_random_quality_correlations(result_dir, replicate_seeds)
    decision = build_decision_table(result_dir, seed_summary, proximity_summary)
    plot_seed_quality(seed_summary, result_dir / "initialization_source_quality")
    plot_proximity_and_correlation(
        proximity_summary,
        correlations,
        result_dir / "initialization_source_influence",
    )
    print(seed_summary.to_string(index=False))
    print(decision.to_string(index=False))
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
