"""Summarize significant k-mers and motif-like families from completed runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from heatshock_kmer_saturation import (
    EXPERIMENT_DIR,
    FRACTIONS,
    kmer_count_enrichment,
)


def main() -> None:
    background = pd.read_csv(EXPERIMENT_DIR / "random_background_1000.csv").sequence.tolist()
    selected = pd.read_csv(EXPERIMENT_DIR / "all_plateau_fraction_sequences.csv")

    count_tables = []
    for (algorithm, fraction), frame in selected.groupby(
        ["algorithm", "plateau_fraction"], sort=True
    ):
        for k in range(1, 7):
            table = kmer_count_enrichment(background, frame.sequence.tolist(), k)
            table.insert(0, "algorithm", algorithm)
            table.insert(1, "plateau_fraction", fraction)
            count_tables.append(table)
    counts = pd.concat(count_tables, ignore_index=True)
    counts.to_csv(EXPERIMENT_DIR / "kmer_count_enrichment_all.csv", index=False)

    presence = pd.read_csv(EXPERIMENT_DIR / "kmer_enrichment_all.csv")
    q90 = presence[
        (presence.plateau_fraction == 0.9)
        & presence.significant
        & presence.k.between(3, 6)
    ].copy()
    consensus = (
        q90.groupby(["k", "kmer"], as_index=False)
        .agg(
            algorithms_enriched=("algorithm", "nunique"),
            algorithms=("algorithm", lambda values: ",".join(sorted(set(values)))),
            best_q_value=("q_value", "min"),
            median_delta_fraction=("delta_fraction", "median"),
            max_delta_fraction=("delta_fraction", "max"),
            max_odds_ratio=("odds_ratio", "max"),
        )
        .sort_values(
            ["algorithms_enriched", "best_q_value", "median_delta_fraction"],
            ascending=[False, True, False],
        )
    )
    consensus.to_csv(EXPERIMENT_DIR / "q90_consensus_enriched_kmers.csv", index=False)

    q90_counts = counts[
        (counts.plateau_fraction == 0.9)
        & counts.significant
        & counts.k.between(3, 6)
    ].copy()
    count_consensus = (
        q90_counts.groupby(["k", "kmer"], as_index=False)
        .agg(
            algorithms_enriched=("algorithm", "nunique"),
            algorithms=("algorithm", lambda values: ",".join(sorted(set(values)))),
            best_q_value=("q_value", "min"),
            median_fold_change=("frequency_fold_change", "median"),
            max_fold_change=("frequency_fold_change", "max"),
        )
        .sort_values(
            ["algorithms_enriched", "best_q_value", "median_fold_change"],
            ascending=[False, True, False],
        )
    )
    count_consensus.to_csv(
        EXPERIMENT_DIR / "q90_consensus_kmer_frequency_increase.csv", index=False
    )

    fraction_summary = (
        presence[presence.significant]
        .groupby(["algorithm", "plateau_fraction", "k"], as_index=False)
        .agg(
            significant_kmers=("kmer", "count"),
            maximum_delta_fraction=("delta_fraction", "max"),
            minimum_q_value=("q_value", "min"),
        )
    )
    fraction_summary.to_csv(
        EXPERIMENT_DIR / "significant_kmer_counts_by_fraction.csv", index=False
    )

    motif_summary = {
        "fractions": list(FRACTIONS),
        "q90_significant_presence_kmers_k3_to_k6": int(len(q90)),
        "q90_significant_frequency_kmers_k3_to_k6": int(len(q90_counts)),
        "q90_presence_kmers_enriched_in_at_least_2_algorithms": int(
            (consensus.algorithms_enriched >= 2).sum()
        ),
        "q90_presence_kmers_enriched_in_at_least_4_algorithms": int(
            (consensus.algorithms_enriched >= 4).sum()
        ),
        "top_consensus_presence_kmers": consensus.head(25).to_dict("records"),
        "top_consensus_frequency_kmers": count_consensus.head(25).to_dict("records"),
    }
    (EXPERIMENT_DIR / "motif_enrichment_summary.json").write_text(
        json.dumps(motif_summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
