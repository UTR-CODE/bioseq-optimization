"""Score source-CV MPRALegNet predictions using Agarwal et al.'s paper protocol.

The source tables contain two *physically assayed* library orientations for
every element.  Figure 3/Table S8 first average the two corresponding model
predictions and then score one value per biological element.  Reporting an
ordinary row-level score instead understates HepG2 Pearson by about 0.03.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from BioSeqBench.evaluation.metrics import compute_metrics


CELL_DATASETS = {
    "hepg2": "agarwal2025_lentimpra_hepg2",
    "k562": "agarwal2025_lentimpra_k562",
    "wtc11": "agarwal2025_lentimpra_wtc11",
}
CELL_NAMES = {"hepg2": "HepG2", "k562": "K562", "wtc11": "WTC11"}


def fold_frame(root: Path, dataset: str, test_fold: int) -> pd.DataFrame:
    reference: pd.DataFrame | None = None
    predictions: list[np.ndarray] = []
    for validation_fold in range(1, 11):
        if validation_fold == test_fold:
            continue
        path = root / dataset / f"model_{validation_fold}_{test_fold}" / "test_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed source-CV prediction: {path}")
        frame = pd.read_csv(path)
        required = {"element_id", "publication_fold", "label_evaluation_averaged", "prediction_tta"}
        if not required <= set(frame):
            raise ValueError(f"Prediction file has an obsolete schema: {path}")
        if not frame["publication_fold"].eq(test_fold).all():
            raise ValueError(f"Unexpected test fold in {path}")
        if reference is None:
            reference = frame.loc[:, ["element_id", "publication_fold", "label_evaluation_averaged"]].copy()
        elif not reference.equals(frame.loc[:, reference.columns]):
            raise ValueError(f"Test row alignment differs between validation models: {path}")
        predictions.append(frame["prediction_tta"].to_numpy(dtype=np.float64))
    if reference is None:
        raise RuntimeError("No validation models collected")
    reference["prediction_tta_ensemble"] = np.mean(np.stack(predictions), axis=0)
    reference["ensemble_members"] = len(predictions)
    return reference


def _paper_table(supplement_root: Path, cell: str) -> pd.DataFrame:
    path = supplement_root / "2023-03-03628-s5" / "2023-03-03628C-Table S8 - folds and performance.xlsx"
    table = pd.read_excel(path, sheet_name=f"Fold assignment, {CELL_NAMES[cell]}")
    table = table.rename(columns={"Fold": "publication_fold", "Sequence ID": "biological_element_id", "Observed log2(RNA/DNA)": "paper_label"})
    required = {"publication_fold", "biological_element_id", "paper_label"}
    if not required <= set(table):
        raise ValueError(f"Unexpected Table S8 columns for {cell}: {list(table.columns)}")
    table["publication_fold"] = pd.to_numeric(table["publication_fold"], errors="raise").astype(int)
    if table["biological_element_id"].duplicated().any():
        raise ValueError(f"Table S8 has duplicated biological IDs for {cell}")
    return table.loc[:, ["publication_fold", "biological_element_id", "paper_label"]]


def _wtc11_id_map(raw_data_root: Path, supplement_root: Path, paper_ids: pd.Series) -> dict[str, str]:
    """Map human_legnet's anonymous WTC11 IDs to Table S8 Ensembl IDs."""
    source = pd.read_csv(raw_data_root / "WTC11_averaged.tsv", sep="\t")
    source = source.loc[pd.to_numeric(source["rev"], errors="raise").eq(0), ["seq_id", "seq"]]
    design_path = supplement_root / "2023-03-03628-s5" / "2023-03-03628C-Table S3 - large-scale lib design (1).xlsx"
    design = pd.read_excel(design_path, sheet_name="WTC11 large-scale")
    sequence_column = "230nt sequence (15nt 5' adaptor - 200nt element - 15nt 3' adaptor)"
    design = design.loc[:, ["name", sequence_column]].rename(columns={sequence_column: "seq"})
    base_name = design["name"].astype(str).str.replace(r"_[FR]$", "", regex=True)
    design = design.loc[base_name.isin(set(paper_ids.astype(str)))].copy()
    if design["seq"].duplicated().any():
        # Technical duplicate rows can occur in Table S3; they must not map a
        # sequence to two distinct biological elements.
        conflicts = design.groupby("seq")["name"].nunique()
        if (conflicts > 1).any():
            raise ValueError("Ambiguous WTC11 Table S3 sequence-to-element mapping")
        design = design.drop_duplicates("seq")
    joined = source.merge(design, on="seq", how="left", validate="one_to_one")
    name = joined["name"].astype(str)
    base = name.str.replace(r"_[FR]$", "", regex=True)
    mapped = joined["name"].notna()
    if not name.loc[mapped].str.endswith(("_F", "_R")).all():
        raise ValueError("Unexpected WTC11 orientation naming in Supplementary Table S3")
    return dict(zip(joined.loc[mapped, "seq_id"].astype(str), base.loc[mapped]))


def _biological_id(oof: pd.DataFrame, cell: str, raw_data_root: Path, supplement_root: Path, paper_ids: pd.Series) -> pd.Series:
    if cell in {"hepg2", "k562"}:
        return oof["element_id"].astype(str).str.replace(r"_Reversed:$", "", regex=True)
    mapping = _wtc11_id_map(raw_data_root, supplement_root, paper_ids)
    return oof["element_id"].astype(str).map(mapping)


def _paper_score(oof: pd.DataFrame, cell: str, raw_data_root: Path, supplement_root: Path) -> tuple[pd.DataFrame, dict[str, float], list[dict[str, object]]]:
    paper = _paper_table(supplement_root, cell)
    paired = oof.loc[:, ["element_id", "publication_fold", "prediction_tta_ensemble"]].copy()
    paired["biological_element_id"] = _biological_id(paired, cell, raw_data_root, supplement_root, paper["biological_element_id"])
    paired = paired.loc[paired["biological_element_id"].notna()].copy()
    counts = paired.groupby(["publication_fold", "biological_element_id"], sort=False).size()
    if (counts > 2).any():
        unexpected = counts.loc[counts.gt(2)].head().to_dict()
        raise ValueError(f"Expected at most two assayed orientations per biological element; examples: {unexpected}")
    single_orientation_elements = int(counts.eq(1).sum())
    paired = paired.groupby(["publication_fold", "biological_element_id"], as_index=False)["prediction_tta_ensemble"].mean()
    paired = paper.merge(paired, on=["publication_fold", "biological_element_id"], how="inner", validate="one_to_one")
    if len(paired) != len(paper):
        # The released WTC11 TSV omits both orientations of ``seq1000`` while
        # Table S8 still lists it.  Score all reproducibly available elements
        # and make this small source-release discrepancy explicit in the log.
        warnings.warn(
            f"Table S8 pairing for {cell} has {len(paper) - len(paired)} element(s) absent from the released TSV; "
            f"scoring {len(paired)} available elements.",
            stacklevel=2,
        )
    metrics = compute_metrics(paired["paper_label"], paired["prediction_tta_ensemble"])
    metrics["single_orientation_elements"] = single_orientation_elements
    metrics["missing_source_elements"] = int(len(paper) - len(paired))
    per_fold = [
        {"test_fold": fold, "paper_test_rows": len(frame), **compute_metrics(frame["paper_label"], frame["prediction_tta_ensemble"])}
        for fold, frame in paired.groupby("publication_fold", sort=True)
    ]
    return paired, metrics, per_fold


def evaluate_cell(root: Path, cell: str, raw_data_root: Path, supplement_root: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    dataset = CELL_DATASETS[cell]
    frames = [fold_frame(root, dataset, fold) for fold in range(1, 11)]
    oof = pd.concat(frames, ignore_index=True)
    if oof["element_id"].duplicated().any():
        raise ValueError(f"Outer folds do not partition {dataset}")
    target = oof["label_evaluation_averaged"].to_numpy(dtype=np.float64)
    prediction = oof["prediction_tta_ensemble"].to_numpy(dtype=np.float64)
    paper_oof, paper_metrics, paper_folds = _paper_score(oof, cell, raw_data_root, supplement_root)
    overall = {
        "dataset": dataset,
        "model": "mpralegnet",
        "cell_type": cell,
        "protocol": "full 10x9 source CV; mean of nine TTA predictions within each outer test fold; average the two physically assayed orientations, then score Table S8 elements",
        "outer_folds": 10,
        "models_per_outer_fold": 9,
        "models_total": 90,
        "oof_construct_rows": len(oof),
        "oof_biological_elements": len(paper_oof),
        **paper_metrics,
        **{f"construct_level_{name}": value for name, value in compute_metrics(target, prediction).items()},
    }
    by_fold = []
    for fold, frame in enumerate(frames, start=1):
        paper_fold = next(item for item in paper_folds if item["test_fold"] == fold)
        by_fold.append({
            "dataset": dataset, "cell_type": cell, "test_fold": fold, "construct_test_rows": len(frame),
            "ensemble_members": 9,
            **{key: value for key, value in paper_fold.items() if key != "test_fold"},
            **{f"construct_level_{name}": value for name, value in compute_metrics(frame["label_evaluation_averaged"], frame["prediction_tta_ensemble"]).items()},
        })
    return overall, oof, paper_oof, by_fold


def evaluate(args: argparse.Namespace) -> list[dict[str, object]]:
    root = Path(args.checkpoint_dir)
    selected = [value.strip().lower() for value in args.cells.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(CELL_DATASETS))
    if unknown:
        raise ValueError(f"Unknown cells: {unknown}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries, folds = [], []
    for cell in selected:
        summary, oof, paper_oof, per_fold = evaluate_cell(root, cell, Path(args.raw_data_root), Path(args.supplement_root))
        summaries.append(summary)
        folds.extend(per_fold)
        oof.to_csv(output.parent / f"{CELL_DATASETS[cell]}_paper_cv_oof_predictions.csv", index=False)
        paper_oof.to_csv(output.parent / f"{CELL_DATASETS[cell]}_paper_orientation_oof_predictions.csv", index=False)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    output.with_suffix(".json").write_text(json.dumps({"summary": summaries, "per_fold": folds}, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate full MPRALegNet source CV predictions")
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints/strong_agarwal2025_mpralegnet_rev0_avg_4ch_v2")
    parser.add_argument("--raw-data-root", default="BioSeqBench/raw_data/agarwal2025_lentimpra/human_legnet")
    parser.add_argument("--supplement-root", default="BioSeqBench/raw_data/agarwal2025_lentimpra")
    parser.add_argument("--cells", default="hepg2,k562,wtc11")
    parser.add_argument("--output", default="BioSeqBench/results/agarwal2025_mpralegnet_paper_cv.csv")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
