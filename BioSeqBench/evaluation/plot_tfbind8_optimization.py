"""Plot TFBind8 optimizer performance for the selected weak models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHM_ORDER = (
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

ALGORITHM_LABELS = (
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

MODEL_ORDER = (0.01, 0.50, 0.95)
MODEL_LABELS = ("Bottom 1%", "Bottom 50%", "Bottom 95%")


def _save_figure(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_activity_distributions(candidates: pd.DataFrame, output_stem: Path) -> None:
    """Draw one candidate-activity distribution panel per weak model."""
    colors = plt.cm.Set2(np.linspace(0, 1, len(ALGORITHM_ORDER)))
    figure, axes = plt.subplots(3, 1, figsize=(13.2, 12.0), constrained_layout=True)

    for axis, fraction, model_label in zip(axes, MODEL_ORDER, MODEL_LABELS):
        model = candidates[np.isclose(candidates["weak_model_fraction"], fraction)]
        values = [
            model.loc[model["algorithm"] == algorithm, "true_score"].to_numpy()
            for algorithm in ALGORITHM_ORDER
        ]
        boxes = axis.boxplot(
            values,
            patch_artist=True,
            widths=0.66,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        rng = np.random.default_rng(31415 + int(fraction * 100))
        for position, observations, color in zip(range(1, 10), values, colors):
            jitter = rng.uniform(-0.18, 0.18, size=len(observations))
            axis.scatter(
                position + jitter,
                observations,
                s=8,
                alpha=0.22,
                color=color,
                edgecolors="none",
                rasterized=True,
            )
            axis.scatter(
                position,
                observations.mean(),
                marker="D",
                s=37,
                color="#b2182b",
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )

        axis.axhline(0.49105, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.set_title(f"Weak model trained on {model_label.lower()} activity sequences", loc="left")
        axis.set_ylabel("Exact TFBind8 activity")
        axis.set_xticks(range(1, 10), ALGORITHM_LABELS, rotation=22, ha="right")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "TFBind8 candidate activity distributions (100 sequences per optimizer)",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.005,
        "Red diamonds: mean activity. Dashed line: global maximum (0.49105).",
        ha="center",
        fontsize=10,
    )
    _save_figure(figure, output_stem)


def _metric_matrix(summary: pd.DataFrame, metric: str) -> np.ndarray:
    matrix = np.empty((len(MODEL_ORDER), len(ALGORITHM_ORDER)), dtype=float)
    for row, fraction in enumerate(MODEL_ORDER):
        for column, algorithm in enumerate(ALGORITHM_ORDER):
            selected = summary[
                np.isclose(summary["weak_model_fraction"], fraction)
                & (summary["algorithm"] == algorithm)
            ]
            if len(selected) != 1:
                raise ValueError(f"Expected one row for fraction={fraction}, algorithm={algorithm}")
            matrix[row, column] = float(selected.iloc[0][metric])
    return matrix


def plot_performance_heatmaps(summary: pd.DataFrame, output_stem: Path) -> None:
    """Draw four complementary summary metrics across all model/optimizer pairs."""
    panels = (
        ("mean_true_score", "Mean activity", ".3f"),
        ("top_10_mean_true_score", "Mean activity of top 10", ".3f"),
        ("best_true_score", "Best activity", ".3f"),
        ("top_1pct_count", "Top-1% hits (out of 100)", ".0f"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 7.8), constrained_layout=True)

    for axis, (metric, title, number_format) in zip(axes.flat, panels):
        matrix = _metric_matrix(summary, metric)
        image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto")
        midpoint = (matrix.min() + matrix.max()) / 2
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                text_color = "white" if value > midpoint else "#222222"
                axis.text(
                    column,
                    row,
                    format(value, number_format),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold" if value == matrix[row].max() else "normal",
                )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(range(len(ALGORITHM_LABELS)), ALGORITHM_LABELS, rotation=30, ha="right")
        axis.set_yticks(range(len(MODEL_LABELS)), MODEL_LABELS)
        axis.tick_params(length=0)
        figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)

    figure.suptitle("TFBind8 optimization performance across three CNN weak models", fontsize=16)
    _save_figure(figure, output_stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="BioSeqBench/results/tfbind8_optimization_benchmark",
    )
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    candidates = pd.read_csv(result_dir / "all_candidates.csv")
    summary = pd.read_csv(result_dir / "summary.csv")
    plot_activity_distributions(candidates, result_dir / "activity_distributions")
    plot_performance_heatmaps(summary, result_dir / "performance_heatmaps")
    print(result_dir / "activity_distributions.png")
    print(result_dir / "performance_heatmaps.png")


if __name__ == "__main__":
    main()
