"""Plot exact-oracle results from the TFBind8 q99-saturation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRACTIONS = (.01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95)
SEEDED = ("ism_greedy", "simulated_annealing", "genetic_algorithm", "adalead", "vanilla_activation_maximization", "fast_seqprop", "cbas", "den")
ALGORITHMS = ("training_top100", "random_screening_median_q99", "random_screening_mean_q99", *SEEDED)
LABEL = {"training_top100": "Train Top-100", "random_screening_median_q99": "Random (median q99)", "random_screening_mean_q99": "Random (mean q99)", "ism_greedy": "ISM", "simulated_annealing": "SA", "genetic_algorithm": "GA", "adalead": "AdaLead", "vanilla_activation_maximization": "Vanilla AM", "fast_seqprop": "Fast SeqProp", "cbas": "CbAS", "den": "DEN"}
COLOR = {"training_top100": "#555555", "random_screening_median_q99": "#999999", "random_screening_mean_q99": "#1b9e77", "ism_greedy": "#e41a1c", "simulated_annealing": "#ff7f00", "genetic_algorithm": "#984ea3", "adalead": "#a65628", "vanilla_activation_maximization": "#f781bf", "fast_seqprop": "#377eb8", "cbas": "#66a61e", "den": "#00a6a6"}


def _save(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _rows(frame: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    return frame[frame.algorithm.eq(algorithm)].set_index("weak_model_fraction").loc[list(FRACTIONS)]


def _style(algorithm: str) -> dict[str, object]:
    if algorithm == "training_top100":
        return {"linestyle": "--", "marker": "s", "linewidth": 1.7}
    if algorithm.startswith("random_screening"):
        return {"linestyle": ":", "marker": "o", "linewidth": 1.8}
    return {"linestyle": "-", "marker": "o", "linewidth": 1.55}


def plot_metrics(aggregate: pd.DataFrame, result_dir: Path) -> None:
    panels = (("mean_true_score", "Mean exact activity of 100 designs"), ("median_true_score", "Median exact activity of 100 designs"), ("top_1pct_count", "Top-1% exact-activity hits (of 100)"), ("best_true_score", "Best exact activity of 100 designs"))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.2), sharex=True, constrained_layout=True)
    x = np.asarray(FRACTIONS) * 100
    for axis, (metric, title) in zip(axes.flat, panels):
        for algorithm in ALGORITHMS:
            rows = _rows(aggregate, algorithm)
            mean = rows[f"{metric}_mean"].to_numpy()
            ci = 1.96 * rows[f"{metric}_sem"].fillna(0).to_numpy()
            axis.plot(x, mean, color=COLOR[algorithm], label=LABEL[algorithm], markersize=4, **_style(algorithm))
            if algorithm in SEEDED:
                axis.fill_between(x, mean - ci, mean + ci, color=COLOR[algorithm], alpha=.065, linewidth=0)
        axis.set_title(title, loc="left", fontweight="bold", fontsize=11)
        axis.set_xticks(x)
        axis.grid(color="#dddddd", linewidth=.7)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[1]:
        axis.set_xlabel("Weak-model lower-tail training fraction (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, .53), frameon=False)
    fig.suptitle("TFBind8 design at each optimizer's q99 saturation point", fontsize=15, fontweight="bold")
    fig.text(.5, -.01, "Points: five-seed mean; bands: 95% normal CI. Scores are the exhaustive experimental TFBind8 values.", ha="center")
    _save(fig, result_dir / "saturation_four_exact_metrics")


def rankings(summary: pd.DataFrame, result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeded = summary[summary.algorithm.isin(SEEDED)].copy()
    seeded["rank"] = seeded.groupby(["weak_model_fraction", "replicate"])["mean_true_score"].rank(ascending=False, method="average")
    seeded["top3"] = seeded["rank"].le(3); seeded["winner"] = seeded["rank"].eq(1)
    overall = seeded.groupby("algorithm", sort=False).agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"), top3_fraction=("top3", "mean"), win_fraction=("winner", "mean"), wins=("winner", "sum"), conditions=("rank", "size"), mean_true_score=("mean_true_score", "mean"), mean_median_true_score=("median_true_score", "mean"), mean_top1pct_hits=("top_1pct_count", "mean"), mean_best_true_score=("best_true_score", "mean"), median_q99_queries=("q99_query_count", "median"), mean_q99_queries=("q99_query_count", "mean")).reset_index().sort_values(["mean_rank", "algorithm"])
    overall.insert(1, "algorithm_label", overall.algorithm.map(LABEL))
    overall.to_csv(result_dir / "saturation_optimizer_ranking.csv", index=False)
    fraction = seeded.groupby(["weak_model_fraction", "algorithm"], sort=False).agg(mean_true_score=("mean_true_score", "mean"), mean_rank=("rank", "mean"), replicate_wins=("winner", "sum")).reset_index()
    fraction["fraction_rank"] = fraction.groupby("weak_model_fraction")["mean_true_score"].rank(ascending=False, method="min")
    fraction["algorithm_label"] = fraction.algorithm.map(LABEL)
    fraction.to_csv(result_dir / "saturation_optimizer_ranking_by_fraction.csv", index=False)
    return overall, fraction


def plot_rank_stability(overall: pd.DataFrame, fraction: pd.DataFrame, result_dir: Path) -> None:
    order = list(overall.algorithm)
    matrix = fraction.pivot(index="weak_model_fraction", columns="algorithm", values="fraction_rank").loc[list(FRACTIONS), order]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.65, 1]}, constrained_layout=True)
    image = axes[0].imshow(matrix.to_numpy(), cmap="RdYlGn_r", vmin=1, vmax=8, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix.iloc[row, column])
            axes[0].text(column, row, value, ha="center", va="center", fontsize=8, fontweight="bold" if value == 1 else "normal", color="white" if value >= 6 else "#222222")
    axes[0].set_title("Rank by training fraction\n(five-seed mean exact activity)", loc="left", fontweight="bold")
    axes[0].set_xticks(range(len(order)), [LABEL[name] for name in order], rotation=35, ha="right")
    axes[0].set_yticks(range(len(FRACTIONS)), [f"{value:g}%" for value in np.asarray(FRACTIONS) * 100])
    axes[0].set_ylabel("Weak-model lower-tail pool"); axes[0].tick_params(length=0)
    fig.colorbar(image, ax=axes[0], fraction=.035, pad=.02, label="Rank (1 is best)")
    ordered = overall.set_index("algorithm").loc[order]
    y = np.arange(len(order)); axes[1].barh(y, ordered.mean_rank, color=[COLOR[name] for name in order])
    axes[1].set_yticks(y, [LABEL[name] for name in order]); axes[1].invert_yaxis(); axes[1].set_xlabel("Mean rank across 60 fraction × seed conditions")
    axes[1].set_title("Global stability", loc="left", fontweight="bold"); axes[1].grid(axis="x", color="#dddddd", linewidth=.7); axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_xlim(0, max(6.2, ordered.mean_rank.max() + 1.2))
    for index, (_, row) in enumerate(ordered.iterrows()):
        axes[1].text(row.mean_rank + .05, index, f"top-3 {row.top3_fraction:.0%}", va="center", fontsize=8)
    fig.suptitle("TFBind8: stable top tier, but no universal optimizer", fontsize=15, fontweight="bold")
    _save(fig, result_dir / "saturation_rank_stability")


def plot_query_cost(summary: pd.DataFrame, result_dir: Path) -> None:
    grouped = summary[summary.algorithm.isin(SEEDED)].groupby(["algorithm", "weak_model_fraction"], sort=False).q99_query_count.agg(["median", "min", "max"]).reset_index()
    fig, axis = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True); x = np.asarray(FRACTIONS) * 100
    for algorithm in SEEDED:
        rows = _rows(grouped.rename(columns={"median": "q99_median"}), algorithm)
        axis.plot(x, rows.q99_median, color=COLOR[algorithm], label=LABEL[algorithm], markersize=4, **_style(algorithm))
        axis.fill_between(x, rows["min"], rows["max"], color=COLOR[algorithm], alpha=.10, linewidth=0)
    axis.set_yscale("log"); axis.set_xticks(x); axis.set_xlabel("Weak-model lower-tail training fraction (%)"); axis.set_ylabel("Weak-model queries to q99 saturation (log scale)")
    axis.set_title("Query cost to reach the observed 99%-of-plateau point", loc="left", fontweight="bold"); axis.grid(color="#dddddd", linewidth=.7, which="both"); axis.spines[["top", "right"]].set_visible(False); axis.legend(frameon=False, ncol=2)
    fig.text(.5, -.01, "Line: median across five seeds; shaded range: min–max. These are weak-CNN calls, not exact-oracle evaluations.", ha="center")
    _save(fig, result_dir / "saturation_query_cost")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="BioSeqBench/results/tfbind8_optimization_saturation_5seeds")
    args = parser.parse_args(); result_dir = Path(args.result_dir)
    summary = pd.read_csv(result_dir / "summary.csv"); aggregate = pd.read_csv(result_dir / "aggregate_summary.csv")
    plot_metrics(aggregate, result_dir); overall, fraction = rankings(summary, result_dir); plot_rank_stability(overall, fraction, result_dir); plot_query_cost(summary, result_dir)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
