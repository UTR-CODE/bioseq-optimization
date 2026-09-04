"""Analyze Train-TOP versus full-space-random seeds in the Optimus benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BioSeqBench.optimization.utr5_benchmark import _sequences_to_encoded


FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SEEDED_ALGORITHMS = (
    "ism_greedy", "simulated_annealing", "genetic_algorithm", "adalead",
    "vanilla_activation_maximization", "fast_seqprop", "cbas", "den",
)
ALGORITHM_LABELS = ("ISM", "SA", "GA", "AdaLead", "Vanilla AM", "Fast SeqProp", "CbAS", "DEN")
SOURCE_LABELS = {"train_top": "Train Top-50", "full_space_random": "Full-space Random-50"}
SOURCE_COLORS = {"train_top": "#2166ac", "full_space_random": "#d6604d"}


def _save(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def build_seed_statistics(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = pd.read_csv(result_dir / "all_shared_initial_sequences.csv")
    rows = []
    for (fraction, replicate, source), group in seeds.groupby(
        ["weak_model_fraction", "replicate", "source"], sort=False
    ):
        weak_best = group.loc[group["weak_prediction"].idxmax()]
        rows.append(
            {
                "weak_model_fraction": fraction,
                "replicate": replicate,
                "source": source,
                "strong_mean": group["strong_score"].mean(),
                "strong_median": group["strong_score"].median(),
                "strong_best": group["strong_score"].max(),
                "weak_mean": group["weak_prediction"].mean(),
                "weak_best": group["weak_prediction"].max(),
                "strong_at_weak_best": weak_best["strong_score"],
                "measured_mean_if_observed": group["measured_label_if_observed"].mean(),
                "in_weak_train_fraction": group["in_weak_train"].mean(),
            }
        )
    replicate = pd.DataFrame(rows)
    metrics = (
        "strong_mean", "strong_median", "strong_best", "weak_mean", "weak_best",
        "strong_at_weak_best", "measured_mean_if_observed", "in_weak_train_fraction",
    )
    named: dict[str, tuple[str, str]] = {}
    for metric in metrics:
        named[f"{metric}_mean"] = (metric, "mean")
        named[f"{metric}_std"] = (metric, "std")
        named[f"{metric}_sem"] = (metric, "sem")
    summary = replicate.groupby(["weak_model_fraction", "source"], sort=False).agg(**named).reset_index()
    replicate.to_csv(result_dir / "initialization_source_replicates.csv", index=False)
    summary.to_csv(result_dir / "initialization_source_summary.csv", index=False)
    return replicate, summary


def build_proximity_statistics(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = pd.read_csv(result_dir / "all_shared_initial_sequences.csv")
    candidates = pd.read_csv(result_dir / "all_candidates.csv")
    candidates = candidates[candidates["algorithm"].isin(SEEDED_ALGORITHMS)]
    rows = []
    for (fraction, replicate, algorithm), group in candidates.groupby(
        ["weak_model_fraction", "replicate", "algorithm"], sort=False
    ):
        condition = seeds[
            np.isclose(seeds["weak_model_fraction"], fraction) & seeds["replicate"].eq(replicate)
        ]
        candidate_encoded = _sequences_to_encoded(group["sequence"].astype(str).to_numpy())
        distances = {}
        for source in ("train_top", "full_space_random"):
            seed_encoded = _sequences_to_encoded(
                condition.loc[condition["source"].eq(source), "sequence"].astype(str).to_numpy()
            )
            distances[source] = (
                candidate_encoded[:, None, :] != seed_encoded[None, :, :]
            ).sum(-1).min(axis=1)
        closer = np.where(
            distances["train_top"] < distances["full_space_random"], "train_top",
            np.where(
                distances["full_space_random"] < distances["train_top"], "full_space_random", "tie"
            ),
        )
        for record, top_distance, random_distance, source in zip(
            group.to_dict("records"), distances["train_top"], distances["full_space_random"], closer
        ):
            rows.append(
                {
                    "weak_model_fraction": fraction,
                    "replicate": replicate,
                    "algorithm": algorithm,
                    "sequence": record["sequence"],
                    "strong_score": record["strong_score"],
                    "minimum_train_top_distance": int(top_distance),
                    "minimum_full_random_distance": int(random_distance),
                    "closer_source": source,
                }
            )
    proximity = pd.DataFrame(rows)
    proximity.to_csv(result_dir / "final_candidate_seed_proximity.csv", index=False)
    summary = proximity.assign(
        closer_train_top=proximity["closer_source"].eq("train_top"),
        closer_full_random=proximity["closer_source"].eq("full_space_random"),
        distance_tie=proximity["closer_source"].eq("tie"),
    ).groupby(["weak_model_fraction", "algorithm"], sort=False).agg(
        closer_train_top_fraction=("closer_train_top", "mean"),
        closer_full_random_fraction=("closer_full_random", "mean"),
        distance_tie_fraction=("distance_tie", "mean"),
        mean_train_top_distance=("minimum_train_top_distance", "mean"),
        mean_full_random_distance=("minimum_full_random_distance", "mean"),
    ).reset_index()
    summary["random_minus_train_proximity"] = (
        summary["closer_full_random_fraction"] - summary["closer_train_top_fraction"]
    )
    summary.to_csv(result_dir / "final_candidate_seed_proximity_summary.csv", index=False)
    return proximity, summary


def build_correlations(result_dir: Path, seed_replicates: pd.DataFrame) -> pd.DataFrame:
    final = pd.read_csv(result_dir / "summary.csv")
    final = final[final["algorithm"].isin(SEEDED_ALGORITHMS)]
    random_stats = seed_replicates[seed_replicates["source"].eq("full_space_random")][
        ["weak_model_fraction", "replicate", "strong_mean", "strong_best"]
    ]
    merged = final.merge(random_stats, on=["weak_model_fraction", "replicate"], validate="many_to_one")
    for column in ("strong_mean", "strong_best"):
        merged[f"{column}_within_model"] = merged[column] - merged.groupby(
            "weak_model_fraction"
        )[column].transform("mean")
    merged["final_within_model"] = merged["mean_strong_score"] - merged.groupby(
        ["algorithm", "weak_model_fraction"]
    )["mean_strong_score"].transform("mean")
    rows = []
    for algorithm, group in merged.groupby("algorithm", sort=False):
        rows.append(
            {
                "algorithm": algorithm,
                "random_seed_mean_vs_final_mean_correlation": group[
                    "strong_mean_within_model"
                ].corr(group["final_within_model"]),
                "random_seed_best_vs_final_mean_correlation": group[
                    "strong_best_within_model"
                ].corr(group["final_within_model"]),
                "conditions": len(group),
            }
        )
    correlations = pd.DataFrame(rows)
    correlations.to_csv(result_dir / "random_seed_quality_correlations.csv", index=False)
    return correlations


def plot_seed_quality(summary: pd.DataFrame, stem: Path) -> None:
    panels = (
        ("strong_mean", "Mean strong-model MRL of 50 seeds", "Strong-model MRL"),
        ("strong_best", "Best strong-model MRL among 50 seeds", "Strong-model MRL"),
        ("strong_at_weak_best", "Strong score of seed ranked best by weak model", "Strong-model MRL"),
        ("in_weak_train_fraction", "Fraction of seeds in weak-model training pool", "Fraction in training"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True, constrained_layout=True)
    x = np.asarray(FRACTIONS) * 100
    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        for source in ("train_top", "full_space_random"):
            rows = summary[summary["source"].eq(source)].set_index("weak_model_fraction").loc[list(FRACTIONS)]
            mean = rows[f"{metric}_mean"].to_numpy()
            interval = 1.96 * rows[f"{metric}_sem"].fillna(0).to_numpy()
            axis.plot(
                x, mean, marker="o" if source == "full_space_random" else "s",
                linewidth=2.1, color=SOURCE_COLORS[source], label=SOURCE_LABELS[source],
            )
            axis.fill_between(x, mean - interval, mean + interval, color=SOURCE_COLORS[source], alpha=0.14, linewidth=0)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.grid(color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Lowest-MRL weak-model training fraction (%)")
    axes[1, 1].set_xlabel("Lowest-MRL weak-model training fraction (%)")
    figure.suptitle("Optimus 5′UTR initialization-source quality", fontsize=16)
    _save(figure, stem)


def plot_influence(proximity: pd.DataFrame, correlations: pd.DataFrame, stem: Path) -> None:
    matrix = np.empty((len(FRACTIONS), len(SEEDED_ALGORITHMS)))
    for row, fraction in enumerate(FRACTIONS):
        for column, algorithm in enumerate(SEEDED_ALGORITHMS):
            selected = proximity[
                np.isclose(proximity["weak_model_fraction"], fraction)
                & proximity["algorithm"].eq(algorithm)
            ]
            matrix[row, column] = selected.iloc[0]["random_minus_train_proximity"] * 100
    limit = max(float(np.abs(matrix).max()), 1)
    figure, axes = plt.subplots(
        1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True
    )
    image = axes[0].imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axes[0].text(
                column, row, f"{value:+.0f}", ha="center", va="center", fontsize=8,
                color="white" if abs(value) > 0.65 * limit else "#222222",
            )
    axes[0].set_title("Final proximity: Random-closer minus Train-TOP-closer (percentage points)", loc="left", fontweight="bold")
    axes[0].set_xticks(range(len(ALGORITHM_LABELS)), ALGORITHM_LABELS, rotation=30, ha="right")
    axes[0].set_yticks(range(len(FRACTIONS)), [f"{value:g}%" for value in np.asarray(FRACTIONS) * 100])
    axes[0].set_ylabel("Weak-model training pool")
    figure.colorbar(image, ax=axes[0], fraction=0.03, pad=0.02)
    ordered = correlations.set_index("algorithm").loc[list(SEEDED_ALGORITHMS)]
    values = ordered["random_seed_mean_vs_final_mean_correlation"].to_numpy()
    axes[1].barh(np.arange(len(values)), values, color=np.where(values >= 0, "#d6604d", "#4393c3"))
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_yticks(range(len(ALGORITHM_LABELS)), ALGORITHM_LABELS)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1, 1)
    axes[1].set_xlabel("Within-model correlation")
    axes[1].set_title("Random-seed mean strong score vs final mean", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#dddddd", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    figure.suptitle("Initialization-source influence (mixed-start evidence, not causal ablation)", fontsize=16)
    _save(figure, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="BioSeqBench/results/utr5_optimization_all_models_5seeds_shared_init",
    )
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    seed_replicates, seed_summary = build_seed_statistics(result_dir)
    _, proximity = build_proximity_statistics(result_dir)
    correlations = build_correlations(result_dir, seed_replicates)
    plot_seed_quality(seed_summary, result_dir / "initialization_source_quality")
    plot_influence(proximity, correlations, result_dir / "initialization_source_influence")
    print(seed_summary.to_string(index=False))
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
