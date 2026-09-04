from __future__ import annotations

import json
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl", **kwargs)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            tabular = [x for x in archive.namelist() if x.lower().endswith((".csv", ".tsv"))]
            if not tabular:
                raise ValueError(f"No CSV/TSV in {path}")
            with archive.open(tabular[0]) as handle:
                return pd.read_csv(handle, sep="\t" if tabular[0].endswith(".tsv") else ",", **kwargs)
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        return pd.read_csv(path, sep="\t", **kwargs)


def pick_column(frame: pd.DataFrame, candidates: tuple[str, ...], *, required: bool = True) -> str | None:
    normalized = {str(column).lower().strip().replace(" ", "_"): column for column in frame.columns}
    for candidate in candidates:
        key = candidate.lower().strip().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"None of columns {candidates} found. Available: {list(frame.columns)}")
    return None


def json_array_mean(value: object) -> float:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return float("nan")
    series = pd.to_numeric(pd.Series(list(value)), errors="coerce")
    return float(series.mean())


class DatasetLoader(ABC):
    name: str
    alphabet: str
    label_type: str
    organism: str
    source_reference: str

    @abstractmethod
    def load(self, raw_path: Path) -> pd.DataFrame:
        """Load raw data to at least sequence/label columns."""

