"""Evaluate Agarwal et al. public MPRALegNet checkpoints without retraining.

The official archive contains nine models for each held-out test fold.  This
script averages these models and their forward/reverse-complement predictions,
then reports both the author-table subset (one assayed orientation per element)
and all source-training rows for an explicit protocol check.
"""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.agarwal2025_lentimpra import OfficialMPRALegNet


CELL_NAMES = {"hepg2": "HepG2", "k562": "K562", "wtc11": "WTC11"}
BASE_CODES = np.full(256, 4, dtype=np.uint8)
for _index, _base in enumerate("AGCT"):
    BASE_CODES[ord(_base)] = _index
COMPLEMENT = str.maketrans("AGCTN", "TCGAN")


def _encode(sequences: pd.Series, *, reverse: bool) -> torch.Tensor:
    encoded = np.zeros((len(sequences), 4, 230), dtype=np.float32)
    for row, sequence in enumerate(sequences.astype(str)):
        if reverse:
            sequence = sequence.translate(COMPLEMENT)[::-1]
        characters = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
        if len(characters) != 230:
            raise ValueError(f"Expected a 230-nt construct, got {len(characters)} nt")
        codes = BASE_CODES[characters]
        known = codes < 4
        encoded[row, codes[known], np.flatnonzero(known)] = 1.0
        encoded[row, :4, ~known] = 0.25
    return torch.from_numpy(encoded)


@torch.inference_mode()
def _predict(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    values: list[np.ndarray] = []
    model.eval()
    for (batch,) in loader:
        values.append(model(batch.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(values)


def _state_dict_from_member(archive: zipfile.ZipFile, member: str) -> dict[str, torch.Tensor]:
    with archive.open(member) as handle:
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    return {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")}


def _table_targets(root: Path, cell_name: str) -> pd.DataFrame:
    table = root / "2023-03-03628-s5" / "2023-03-03628C-Table S8 - folds and performance.xlsx"
    frame = pd.read_excel(table, sheet_name=f"Fold assignment, {cell_name}")
    return frame.rename(columns={"Fold": "fold", "Sequence ID": "seq_id", "Observed log2(RNA/DNA)": "paper_label"})


def _load_rows(data_root: Path, cell_name: str) -> pd.DataFrame:
    frame = pd.read_csv(data_root / f"{cell_name}_averaged.tsv", sep="\t")
    frame = frame.loc[pd.to_numeric(frame["rev"], errors="raise").eq(0)].copy()
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    if frame["seq_id"].duplicated().any():
        raise ValueError(f"{cell_name} has duplicate rev=0 sequence IDs")
    return frame


def _members(archive: zipfile.ZipFile, cell_name: str, test_fold: int) -> list[str]:
    pattern = re.compile(rf"^best_model/models/{cell_name}/md_shift_reverse_noavg_noch/model_test{test_fold}_val([1-9]|10)\.ckpt$")
    members = sorted(member for member in archive.namelist() if pattern.match(member))
    if len(members) != 9:
        raise ValueError(f"Expected nine checkpoints for {cell_name} test fold {test_fold}, found {len(members)}")
    return members


def evaluate_cell(archive: zipfile.ZipFile, data_root: Path, supplement_root: Path, cell: str, device: torch.device, batch_size: int) -> tuple[list[dict[str, object]], pd.DataFrame]:
    cell_name = CELL_NAMES[cell]
    rows = _load_rows(data_root, cell_name)
    paper = _table_targets(supplement_root, cell_name)
    oof: list[pd.DataFrame] = []
    per_fold: list[dict[str, object]] = []
    for test_fold in range(1, 11):
        test = rows.loc[rows["fold"].eq(test_fold)].reset_index(drop=True)
        forward, reverse = _encode(test["seq"], reverse=False), _encode(test["seq"], reverse=True)
        predictions: list[np.ndarray] = []
        for member in _members(archive, cell_name, test_fold):
            model = OfficialMPRALegNet().to(device)
            missing, unexpected = model.load_state_dict(_state_dict_from_member(archive, member), strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Checkpoint incompatibility in {member}: missing={missing}, unexpected={unexpected}")
            predictions.append((_predict(model, forward, device, batch_size) + _predict(model, reverse, device, batch_size)) / 2.0)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        test["prediction_tta_ensemble"] = np.mean(np.stack(predictions), axis=0)
        paper_test = paper.loc[paper["fold"].eq(test_fold)].copy()
        predictions_by_id = test.set_index("seq_id")["prediction_tta_ensemble"]
        paper_test["prediction_single_orientation"] = paper_test["seq_id"].map(predictions_by_id)
        # HepG2 and K562 Table S8 scores are computed after averaging the two
        # *physically assayed* library orientations (not merely an RC-TTA pass).
        # Both rows have one identical activity label and paired IDs differ only
        # by this source-table suffix.
        paired_id = paper_test["seq_id"].astype(str) + "_Reversed:"
        paper_test["prediction_paired_orientation"] = (
            paper_test["prediction_single_orientation"].to_numpy() + paired_id.map(predictions_by_id).to_numpy()
        ) / 2.0
        if paper_test[["prediction_single_orientation", "prediction_paired_orientation"]].isna().any().any():
            raise ValueError(
                f"Table S8 orientation-pair row mismatch for {cell_name} fold {test_fold}; "
                "this source table does not expose the required pair IDs."
            )
        per_fold.append({
            "cell_type": cell_name, "test_fold": test_fold, "rows_author_table": len(paper_test), "rows_all_orientations": len(test),
            **{f"author_table_{key}": value for key, value in compute_metrics(paper_test["paper_label"], paper_test["prediction_paired_orientation"]).items()},
            **{f"author_table_single_orientation_{key}": value for key, value in compute_metrics(paper_test["paper_label"], paper_test["prediction_single_orientation"]).items()},
            **{f"all_orientations_{key}": value for key, value in compute_metrics(test["mean_value"], test["prediction_tta_ensemble"]).items()},
        })
        oof.append(test)
    oof_frame = pd.concat(oof, ignore_index=True)
    predictions_by_id = oof_frame.set_index("seq_id")["prediction_tta_ensemble"]
    paper_oof = paper.copy()
    paper_oof["prediction_single_orientation"] = paper_oof["seq_id"].map(predictions_by_id)
    paired_id = paper_oof["seq_id"].astype(str) + "_Reversed:"
    paper_oof["prediction_paired_orientation"] = (
        paper_oof["prediction_single_orientation"].to_numpy() + paired_id.map(predictions_by_id).to_numpy()
    ) / 2.0
    if paper_oof[["prediction_single_orientation", "prediction_paired_orientation"]].isna().any().any():
        raise ValueError(f"Table S8 orientation-pair row mismatch for {cell_name}")
    summary = {
        "cell_type": cell_name, "models_total": 90, "protocol": "official checkpoints; 9-model ensemble per held-out fold; forward/reverse TTA",
        "rows_author_table": len(paper_oof), "rows_all_orientations": len(oof_frame),
        **{f"author_table_pooled_{key}": value for key, value in compute_metrics(paper_oof["paper_label"], paper_oof["prediction_paired_orientation"]).items()},
        **{f"author_table_mean_fold_{key}": float(np.mean([row[f"author_table_{key}"] for row in per_fold])) for key in ("pearson", "spearman", "mse", "r2")},
        **{f"all_orientations_pooled_{key}": value for key, value in compute_metrics(oof_frame["mean_value"], oof_frame["prediction_tta_ensemble"]).items()},
    }
    return [summary, *per_fold], oof_frame


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    selected = [value.strip().lower() for value in args.cells.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(CELL_NAMES))
    if unknown:
        raise ValueError(f"Unknown cells: {unknown}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    with zipfile.ZipFile(args.archive) as archive:
        for cell in selected:
            result, oof = evaluate_cell(archive, Path(args.data_root), Path(args.supplement_root), cell, device, args.batch_size)
            summaries.extend(result)
            oof.to_csv(output.parent / f"official_agarwal2025_{CELL_NAMES[cell]}_oof.csv", index=False)
    fields = sorted({key for row in summaries for key in row})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="tmp/best_model.zip")
    parser.add_argument("--data-root", default="BioSeqBench/raw_data/agarwal2025_lentimpra/human_legnet")
    parser.add_argument("--supplement-root", default="BioSeqBench/raw_data/agarwal2025_lentimpra")
    parser.add_argument("--cells", default="hepg2,k562,wtc11")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", default="BioSeqBench/results/agarwal2025_official_weights.csv")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
