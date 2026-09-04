"""Visualize the 12-model, five-seed Optimus 5'UTR design benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = (
    "training_top100", "random_screening", "ism_greedy", "simulated_annealing",
    "genetic_algorithm", "adalead", "vanilla_activation_maximization",
    "fast_seqprop", "cbas", "den",
)
LABELS = (
    "Train Top-100", "Random", "ISM", "SA", "GA", "AdaLead",
    "Vanilla AM", "Fast SeqProp", "CbAS", "DEN",
)
FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def _save(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    matrix = np.empty((len(FRACTIONS), len(ALGORITHMS)))
    for row, fraction in enumerate(FRACTIONS):
        for column_index, algorithm in enumerate(ALGORITHMS):
            selected = frame[
                np.isclose(frame["weak_model_fraction"], fraction)
                & (frame["algorithm"] == algorithm)
            ]
            if len(selected) != 1:
                raise ValueError(f"Expected one row for {fraction=} and {algorithm=}")
            matrix[row, column_index] = float(selected.iloc[0][column])
    return matrix


def _annotated_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    title: str,
    cmap: str,
    number_format: str,
) -> None:
    image = axis.imshow(matrix, aspect="auto", cmap=cmap)
    normalized = image.norm(matrix)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column, row, format(value, number_format), ha="center", va="center",
                fontsize=7.2, color="white" if normalized[row, column] > 0.70 else "#222222",
                fontweight="bold" if value == matrix[row].max() else "normal",
            )
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    axis.set_yticks(range(len(FRACTIONS)), [f"{value:g}%" for value in np.asarray(FRACTIONS) * 100])
    axis.set_ylabel("Weak-model lower-tail pool")
    axis.tick_params(length=0)
    axis.figure.colorbar(image, ax=axis, fraction=0.018, pad=0.015)


def plot_four_metric_curves(aggregate: pd.DataFrame, output_stem: Path) -> None:
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(ALGORITHMS)))
    markers = ("*", "o", "s", "^", "D", "P", "X", "v", "<", ">")
    panels = (
        ("mean_strong_score", "Mean strong-model MRL of 100 designs"),
        ("median_strong_score", "Median strong-model MRL of 100 designs"),
        ("top_1pct_count", "Top-1% hits among 100 designs"),
        ("best_strong_score", "Best strong-model MRL among 100 designs"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 10.5), sharex=True, constrained_layout=True)
    x = np.asarray(FRACTIONS) * 100
    for axis, (metric, title) in zip(axes.flat, panels):
        for algorithm, label, color, marker in zip(ALGORITHMS, LABELS, colors, markers):
            rows = aggregate[aggregate["algorithm"] == algorithm].set_index(
                "weak_model_fraction"
            ).loc[list(FRACTIONS)]
            mean = rows[f"{metric}_mean"].to_numpy()
            interval = 1.96 * rows[f"{metric}_sem"].fillna(0).to_numpy()
            axis.plot(
                x, mean, label=label, color=color, marker=marker, markersize=4.5,
                linewidth=1.7, linestyle="--" if algorithm == "training_top100" else "-",
            )
            if algorithm != "training_top100":
                axis.fill_between(x, mean - interval, mean + interval, color=color, alpha=0.07, linewidth=0)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(title)
        axis.set_xticks(x)
        axis.grid(color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, bbox_to_anchor=(1.01, 0.92), loc="upper left")
    axes[1, 0].set_xlabel("Lowest-MRL training fraction (%)")
    axes[1, 1].set_xlabel("Lowest-MRL training fraction (%)")
    figure.suptitle("Optimus 5′UTR design performance (5-seed mean)", fontsize=16, fontweight="bold")
    figure.text(0.5, -0.01, "Shading: 95% CI. All generated candidates are scored only by the frozen strong Optimus model.", ha="center")
    _save(figure, output_stem)


def plot_four_metric_heatmaps(aggregate: pd.DataFrame, output_stem: Path) -> None:
    panels = (
        ("mean_strong_score_mean", "Mean strong-model MRL", ".2f", "YlGnBu"),
        ("median_strong_score_mean", "Median strong-model MRL", ".2f", "YlGnBu"),
        ("top_1pct_count_mean", "Top-1% hits (out of 100)", ".1f", "YlOrRd"),
        ("best_strong_score_mean", "Best strong-model MRL", ".2f", "YlGnBu"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(19, 13), constrained_layout=True)
    for axis, (metric, title, number_format, cmap) in zip(axes.flat, panels):
        _annotated_heatmap(
            axis, _matrix(aggregate, metric), title=title, cmap=cmap, number_format=number_format
        )
    figure.suptitle("Optimus 5′UTR optimization across all 12 weak models", fontsize=17, fontweight="bold")
    _save(figure, output_stem)


def write_overall_ranking(summary: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    ranked = summary[summary["algorithm"] != "training_top100"].copy()
    ranked["rank"] = ranked.groupby(["weak_model_fraction", "replicate"])[
        "mean_strong_score"
    ].rank(method="average", ascending=False)
    ranked["is_winner"] = ranked["rank"].eq(1)
    result = ranked.groupby("algorithm", sort=False).agg(
        mean_rank=("rank", "mean"),
        median_rank=("rank", "median"),
        wins=("is_winner", "sum"),
        conditions=("rank", "size"),
        mean_strong_score=("mean_strong_score", "mean"),
        mean_top_1pct_hits=("top_1pct_count", "mean"),
        mean_best_strong_score=("best_strong_score", "mean"),
    ).reset_index().sort_values(["mean_rank", "algorithm"])
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="BioSeqBench/results/utr5_optimization_all_models_5seeds_shared_init",
    )
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    summary = pd.read_csv(result_dir / "summary.csv")
    aggregate = pd.read_csv(result_dir / "aggregate_summary.csv")
    plot_four_metric_curves(aggregate, result_dir / "four_metric_curves")
    plot_four_metric_heatmaps(aggregate, result_dir / "four_metric_heatmaps")
    ranking = write_overall_ranking(summary, result_dir / "overall_algorithm_ranking.csv")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
