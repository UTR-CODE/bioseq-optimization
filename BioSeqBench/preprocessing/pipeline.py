from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .common import clean_table, deterministic_split, write_dataset
from .loaders import LOADERS
from .qc import dataset_statistics, write_html_report


def prepare_dataset(
    dataset_name: str,
    raw_path: str | Path,
    output_root: str | Path = "BioSeqBench/dataset",
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    report_path: str | Path = "BioSeqBench/reports/dataset_statistics.html",
) -> dict:
    key = dataset_name.lower()
    if key not in LOADERS:
        raise KeyError(f"Unknown dataset {dataset_name!r}; choose from {sorted(LOADERS)}")
    loader = LOADERS[key]()
    raw = loader.load(Path(raw_path))
    cleaned, audit = clean_table(
        raw,
        dataset_name=loader.name,
        alphabet=loader.alphabet,
        label_type=loader.label_type,
        organism=loader.organism,
        source_reference=loader.source_reference,
    )
    if cleaned.empty:
        raise ValueError(f"No valid rows remain for {dataset_name}")
    if "source_split" in cleaned and set(cleaned["source_split"].unique()) >= {"train", "test"}:
        splits = {
            name: cleaned.loc[cleaned["source_split"] == name].reset_index(drop=True)
            for name in ("train", "val", "test")
        }
        split_method = (
            "original publication split"
            if len(splits["val"])
            else "original publication split (no validation set was used)"
        )
    else:
        group_column = "split_group" if "split_group" in cleaned else "sequence"
        splits = deterministic_split(
            cleaned, seed=seed, train_fraction=train_fraction, val_fraction=val_fraction,
            group_column=group_column,
        )
        split_method = f"SHA-256({group_column}, seed), grouped by {group_column}"
    coverage = getattr(loader, "coverage_check", None)
    if key in {"tfbind8", "tfbind10"}:
        sequence_length = 8 if key == "tfbind8" else 10
        coverage = {
            "expected_sequences": 4**sequence_length,
            "observed_unique_sequences": int(cleaned.sequence.nunique()),
            f"complete_4_pow_{sequence_length}_landscape": int(cleaned.sequence.nunique()) == 4**sequence_length,
        }
        if key == "tfbind8":
            coverage.update({
                "reverse_complement_equivalence_groups": int(cleaned.split_group.nunique()),
                "reverse_complement_pairs_kept_together": True,
            })
    stats = dataset_statistics(cleaned, audit)
    metadata = {
        "dataset_name": loader.name,
        "alphabet": loader.alphabet,
        "label_type": loader.label_type,
        "organism": loader.organism,
        "source_reference": loader.source_reference,
        "raw_path": str(Path(raw_path).resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split_method": split_method,
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
        "canonical_columns": ["sequence_id", "sequence", "alphabet", "length", "label", "label_type", "cell_type", "organism", "source_reference"],
        "target_columns": (
            [column for column in cleaned if column.startswith("label_") and column != "label_type"]
            or ["label"]
        ),
        "target_descriptions": getattr(loader, "target_descriptions", None),
        "cleaning_audit": audit,
        "coverage_check": coverage,
        "statistics": stats,
    }
    metadata.update(getattr(loader, "metadata_extras", {}))
    write_dataset(splits, Path(output_root) / key, metadata)

    report = Path(report_path)
    sidecar = report.with_suffix(".json")
    existing = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    existing[key] = stats
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
    write_html_report(existing, report)
    return metadata
