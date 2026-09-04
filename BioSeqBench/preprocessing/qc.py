"""Dependency-light HTML quality-control report generation."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _describe(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {key: float(value) for key, value in numeric.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).items()}


def dataset_statistics(frame: pd.DataFrame, audit: dict) -> dict:
    lengths = frame["length"]
    labels = [
        column for column in frame.columns
        if column == "label" or (column.startswith("label_") and column != "label_type")
    ]
    alphabet = str(frame["alphabet"].iloc[0]) if len(frame) else ""
    kmer_counts: Counter[str] = Counter()
    k = 2
    for seq in frame["sequence"].astype(str):
        kmer_counts.update(seq[i:i + k] for i in range(max(0, len(seq) - k + 1)))
    return {
        **audit,
        "number_of_sequences": int(len(frame)),
        "unique_sequences": int(frame["sequence"].nunique()),
        "alphabet": alphabet,
        "length_distribution": _describe(lengths),
        "label_distribution": {column: _describe(frame[column]) for column in labels},
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "gc_content_distribution": _describe(frame["gc_content"]) if "gc_content" in frame else {},
        "sequence_entropy_distribution": _describe(frame["sequence_entropy"]),
        "aggregate_2mer_frequency": {
            key: count / max(1, sum(kmer_counts.values())) for key, count in sorted(kmer_counts.items())
        },
    }


def _dict_table(title: str, data: dict) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(json.dumps(value, ensure_ascii=False))}</code></td></tr>"
        for key, value in data.items()
    )
    return f"<h3>{html.escape(title)}</h3><table>{rows}</table>"


def write_html_report(all_statistics: dict[str, dict], output: Path) -> None:
    sections = []
    for name, stats in sorted(all_statistics.items()):
        primary = {key: value for key, value in stats.items() if not isinstance(value, dict)}
        detail = {key: value for key, value in stats.items() if isinstance(value, dict)}
        body = _dict_table("Summary", primary) + "".join(_dict_table(key, value) for key, value in detail.items())
        sections.append(f"<section><h2>{html.escape(name)}</h2>{body}</section>")
    document = """<!doctype html><html><head><meta charset='utf-8'>
    <title>BioSeqBench dataset statistics</title><style>
    body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a}
    section{margin:2rem 0;padding:1.2rem;border:1px solid #d5d8dc;border-radius:8px}
    table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}th,td{border:1px solid #ddd;padding:.45rem;text-align:left}
    th{width:30%;background:#f4f6f7}code{white-space:pre-wrap;word-break:break-word}</style></head><body>
    <h1>BioSeqBench QC report</h1>""" + "".join(sections) + "</body></html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
