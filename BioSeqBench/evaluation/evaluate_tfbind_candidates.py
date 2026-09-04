"""Score generated TFBind8/TFBind10 candidates with the exact experimental oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from BioSeqBench.oracles.tfbind import encode_sequences, encoded_to_codes, load_tfbind_landscape


def evaluate_candidates(
    dataset: str,
    sequences: list[str],
    *,
    train_fraction: float | None = None,
    raw_path: str | Path | None = None,
    top_k: int = 100,
) -> tuple[pd.DataFrame, dict[str, float | int | bool | str | None]]:
    """Return per-candidate true scores and optimization-focused summary metrics."""
    if not sequences:
        raise ValueError("Candidate list is empty")
    if train_fraction is not None and not 0 < train_fraction <= 0.95:
        raise ValueError("train_fraction must be in (0, 0.95]")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    landscape = load_tfbind_landscape(dataset, raw_path)
    normalized = [str(sequence).upper() for sequence in sequences]
    codes = encoded_to_codes(encode_sequences(normalized, landscape.length))
    true_scores = landscape.score_codes(codes)

    ranked_codes = landscape.ranked_codes()
    rank_by_code = np.empty(landscape.size, dtype=np.int64)
    rank_by_code[ranked_codes] = np.arange(landscape.size)
    activity_percentile = 100.0 * (rank_by_code[codes] + 1) / landscape.size
    in_weak_train = np.zeros(len(codes), dtype=bool)
    train_count = None
    if train_fraction is not None:
        train_count = int(round(landscape.size * train_fraction))
        training_mask = np.zeros(landscape.size, dtype=bool)
        training_mask[ranked_codes[:train_count]] = True
        in_weak_train = training_mask[codes]

    scored = pd.DataFrame(
        {
            "sequence": normalized,
            "sequence_code": codes,
            "true_score": true_scores,
            "activity_percentile": activity_percentile,
            "in_weak_train": in_weak_train,
        }
    )
    unique = scored.drop_duplicates("sequence_code")
    ordered = unique.sort_values("true_score", ascending=False)
    best = float(ordered.iloc[0]["true_score"])
    best_percentile = float(ordered.iloc[0]["activity_percentile"])
    top = ordered.head(min(top_k, len(ordered)))
    novel = ordered.loc[~ordered["in_weak_train"]] if train_fraction is not None else ordered
    summary: dict[str, float | int | bool | str | None] = {
        "dataset": landscape.dataset,
        "train_fraction": train_fraction,
        "weak_train_count": train_count,
        "candidate_count": len(scored),
        "unique_candidate_count": len(unique),
        "unique_fraction": len(unique) / len(scored),
        "novel_unique_count": len(novel),
        "novel_unique_fraction": len(novel) / len(unique),
        "best_true_score": best,
        "best_activity_percentile": best_percentile,
        "mean_true_score": float(unique["true_score"].mean()),
        f"top_{top_k}_mean_true_score": float(top["true_score"].mean()),
        "best_novel_true_score": float(novel.iloc[0]["true_score"]) if len(novel) else None,
        "best_novel_activity_percentile": float(novel.iloc[0]["activity_percentile"]) if len(novel) else None,
        "global_optimum_score": float(landscape.labels_by_code.max()),
        "global_optimum_found": bool(np.isclose(best, landscape.labels_by_code.max(), rtol=0, atol=1e-12)),
    }
    return scored, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["tfbind8", "tfbind10"])
    parser.add_argument("--candidates", required=True, help="CSV containing generated sequences")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--train-fraction", type=float, help="Weak-model lower-tail fraction, for novelty metrics")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--raw-path")
    parser.add_argument("--output-dir", default="BioSeqBench/results/tfbind_optimization")
    parser.add_argument("--run-name", help="Output stem; defaults to the candidate CSV stem")
    args = parser.parse_args()

    candidate_path = Path(args.candidates)
    frame = pd.read_csv(candidate_path)
    if args.sequence_column not in frame:
        raise ValueError(f"Column {args.sequence_column!r} not found in {candidate_path}")
    scored, summary = evaluate_candidates(
        args.dataset,
        frame[args.sequence_column].astype(str).tolist(),
        train_fraction=args.train_fraction,
        raw_path=args.raw_path,
        top_k=args.top_k,
    )
    # Preserve algorithm-specific columns alongside canonical exact-oracle fields.
    for column in frame.columns:
        if column not in scored and column != args.sequence_column:
            scored[column] = frame[column].to_numpy()
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or candidate_path.stem
    scored.to_csv(output_dir / f"{run_name}_scored.csv", index=False)
    (output_dir / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
