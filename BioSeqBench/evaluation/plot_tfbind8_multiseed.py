"""Visualize the all-model, multi-seed TFBind8 optimization benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = (
    "training_top100",
    "random_screening",
    "ism_greedy",
    "simulated_annealing",
    "genetic_algorithm",
    "adalead",
    "vanilla_activation_maximization",
    "fast_seqprop",
    "cbas",
    "den",
)
LABELS = (
    "Train Top-100",
    "Random",
    "ISM",
    "SA",
    "GA",
    "AdaLead",
    "Vanilla AM",
    "Fast SeqProp",
    "CbAS",
    "DEN",
)
FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def _save(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    output = np.empty((len(FRACTIONS), len(ALGORITHMS)), dtype=float)
    for row, fraction in enumerate(FRACTIONS):
        for col, algorithm in enumerate(ALGORITHMS):
            selected = frame[
                np.isclose(frame["weak_model_fraction"], fraction)
                & (frame["algorithm"] == algorithm)
            ]
            if len(selected) != 1:
                raise ValueError(f"Expected one aggregate row for {fraction=} and {algorithm=}")
            output[row, col] = float(selected.iloc[0][column])
    return output


def plot_learning_curves(aggregate: pd.DataFrame, output_stem: Path) -> None:
    """Plot performance as the lower-tail training-pool fraction grows."""
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(ALGORITHMS)))
    markers = ("*", "o", "s", "^", "D", "P", "X", "v", "<", ">")
    replicate_count = int(aggregate["replicates"].max())
    figure, axes = plt.subplots(2, 1, figsize=(13.5, 10.0), sharex=True, constrained_layout=True)
    specifications = (
        ("mean_true_score", "Mean exact activity of 100 designs"),
        ("top_1pct_count", "Top-1% sequences among 100 designs"),
    )

    for axis, (metric, ylabel) in zip(axes, specifications):
        for algorithm, label, color, marker in zip(ALGORITHMS, LABELS, colors, markers):
            rows = aggregate[aggregate["algorithm"] == algorithm].copy()
            rows = rows.set_index("weak_model_fraction").loc[list(FRACTIONS)].reset_index()
            mean = rows[f"{metric}_mean"].to_numpy()
            interval = 1.96 * rows[f"{metric}_sem"].fillna(0).to_numpy()
            axis.plot(
                np.asarray(FRACTIONS) * 100,
                mean,
                label=label,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.8,
            )
            axis.fill_between(
                np.asarray(FRACTIONS) * 100,
                mean - interval,
                mean + interval,
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        axis.set_ylabel(ylabel)
        axis.grid(color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("TFBind8 optimizer performance across weak-model training pools", loc="left", fontweight="bold")
    axes[0].legend(ncol=3, frameon=False, bbox_to_anchor=(1.0, 1.02), loc="upper left")
    axes[1].set_xlabel("Lowest-activity fraction used to train the weak model (%)")
    axes[1].set_xticks(np.asarray(FRACTIONS) * 100)
    figure.text(
        0.5,
        -0.01,
        f"Lines: mean over {replicate_count} seeds. Shading: 95% confidence interval. "
        "Train Top-100 is deterministic.",
        ha="center",
    )
    _save(figure, output_stem)


def _annotated_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    title: str,
    cmap: str,
    number_format: str,
    symmetric: bool = False,
) -> None:
    limit = float(np.abs(matrix).max()) if symmetric else None
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=-limit if symmetric else None,
        vmax=limit if symmetric else None,
    )
    normalized = image.norm(matrix)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            intensity = normalized[row, col]
            use_white = (
                (symmetric and (intensity < 0.18 or intensity > 0.82))
                or (not symmetric and intensity > 0.70)
            )
            color = "white" if use_white else "#222222"
            axis.text(
                col,
                row,
                format(value, number_format),
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
                fontweight="bold" if value == matrix[row].max() else "normal",
            )
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    axis.set_yticks(range(len(FRACTIONS)), [f"{fraction:g}%" for fraction in np.asarray(FRACTIONS) * 100])
    axis.set_ylabel("Weak-model training pool")
    axis.tick_params(length=0)
    axis.figure.colorbar(image, ax=axis, fraction=0.018, pad=0.015)


def plot_heatmaps(aggregate: pd.DataFrame, output_stem: Path) -> None:
    """Plot absolute activity and paired improvement over random screening."""
    mean_activity = _matrix(aggregate, "mean_true_score_mean")
    gain = _matrix(aggregate, "mean_activity_gain_over_random_mean")
    replicate_count = int(aggregate["replicates"].max())
    figure, axes = plt.subplots(2, 1, figsize=(15.5, 12.5), constrained_layout=True)
    _annotated_heatmap(
        axes[0],
        mean_activity,
        title=f"Mean exact activity ({replicate_count}-seed mean; Train Top-100 is deterministic)",
        cmap="YlGnBu",
        number_format=".3f",
    )
    _annotated_heatmap(
        axes[1],
        gain,
        title="Mean activity gain over matched random screening",
        cmap="RdBu_r",
        number_format="+.3f",
        symmetric=True,
    )
    figure.suptitle("TFBind8 all-model optimization benchmark", fontsize=16)
    _save(figure, output_stem)


def plot_four_metric_curves(aggregate: pd.DataFrame, output_stem: Path) -> None:
    """Plot mean, median, top-1% hits, and best activity for every method."""
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(ALGORITHMS)))
    markers = ("*", "o", "s", "^", "D", "P", "X", "v", "<", ">")
    panels = (
        ("mean_true_score", "Mean activity of 100 designs"),
        ("median_true_score", "Median activity of 100 designs"),
        ("top_1pct_count", "Top-1% hits among 100 designs"),
        ("best_true_score", "Best activity among 100 designs"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 10.5), sharex=True, constrained_layout=True)
    x_values = np.asarray(FRACTIONS) * 100

    for axis, (metric, ylabel) in zip(axes.flat, panels):
        for algorithm, label, color, marker in zip(ALGORITHMS, LABELS, colors, markers):
            rows = aggregate[aggregate["algorithm"] == algorithm]
            rows = rows.set_index("weak_model_fraction").loc[list(FRACTIONS)].reset_index()
            mean = rows[f"{metric}_mean"].to_numpy()
            interval = 1.96 * rows[f"{metric}_sem"].fillna(0).to_numpy()
            axis.plot(
                x_values,
                mean,
                label=label,
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.7,
                linestyle="--" if algorithm == "training_top100" else "-",
            )
            if algorithm != "training_top100":
                axis.fill_between(
                    x_values,
                    mean - interval,
                    mean + interval,
                    color=color,
                    alpha=0.07,
                    linewidth=0,
                )
        axis.set_title(ylabel, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_values)
        axis.grid(color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=1,
        frameon=False,
        bbox_to_anchor=(1.01, 0.92),
        loc="upper left",
    )
    axes[1, 0].set_xlabel("Weak-model lower-tail training fraction (%)")
    axes[1, 1].set_xlabel("Weak-model lower-tail training fraction (%)")
    figure.suptitle(
        "TFBind8 optimizer performance: four complementary metrics (5-seed mean)",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        "Shading: 95% confidence interval. Train Top-100 is deterministic and shown with dashed lines.",
        ha="center",
    )
    _save(figure, output_stem)


def plot_four_metric_heatmaps(aggregate: pd.DataFrame, output_stem: Path) -> None:
    """Draw annotated all-model heatmaps for the four requested metrics."""
    panels = (
        ("mean_true_score_mean", "Mean activity", ".3f", "YlGnBu"),
        ("median_true_score_mean", "Median activity", ".3f", "YlGnBu"),
        ("top_1pct_count_mean", "Top-1% hits (out of 100)", ".1f", "YlOrRd"),
        ("best_true_score_mean", "Best activity", ".3f", "YlGnBu"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(19.0, 13.0), constrained_layout=True)
    for axis, (metric, title, number_format, cmap) in zip(axes.flat, panels):
        _annotated_heatmap(
            axis,
            _matrix(aggregate, metric),
            title=title,
            cmap=cmap,
            number_format=number_format,
        )
    figure.suptitle(
        "TFBind8 optimization metrics across all weak models (5-seed mean)",
        fontsize=17,
        fontweight="bold",
    )
    _save(figure, output_stem)


def write_overall_ranking(summary: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Rank methods within each weak-model/seed condition by mean activity."""
    ranked = summary[summary["algorithm"] != "training_top100"].copy()
    ranked["rank"] = ranked.groupby(["weak_model_fraction", "replicate"])[
        "mean_true_score"
    ].rank(method="average", ascending=False)
    ranked["is_winner"] = ranked["rank"] == 1
    overall = (
        ranked.groupby("algorithm", sort=False)
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            wins=("is_winner", "sum"),
            conditions=("rank", "size"),
            mean_activity=("mean_true_score", "mean"),
            mean_top_1pct_hits=("top_1pct_count", "mean"),
        )
        .reset_index()
        .sort_values(["mean_rank", "algorithm"])
    )
    overall.to_csv(output_path, index=False)
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="BioSeqBench/results/tfbind8_optimization_all_models_5seeds",
    )
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    summary = pd.read_csv(result_dir / "summary.csv")
    aggregate = pd.read_csv(result_dir / "aggregate_summary.csv")
    plot_learning_curves(aggregate, result_dir / "multiseed_learning_curves")
    plot_heatmaps(aggregate, result_dir / "multiseed_heatmaps")
    plot_four_metric_curves(aggregate, result_dir / "four_metric_curves")
    plot_four_metric_heatmaps(aggregate, result_dir / "four_metric_heatmaps")
    ranking = write_overall_ranking(summary, result_dir / "overall_algorithm_ranking.csv")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
