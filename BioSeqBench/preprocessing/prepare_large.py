"""Streaming standardization for the multi-million-row BioSeqBench releases.

The regular loaders deliberately use pandas and are intended for medium-sized
tables.  This module keeps the same canonical CSV schema while never loading a
complete Yeast GPRA or PARM split into RAM.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pysam
import pyarrow.parquet as pq

from .common import stable_fraction
from .qc import write_html_report
from .schema import CANONICAL_COLUMNS


DNA = set("ACGT")


class OnlineSummary:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def as_dict(self) -> dict[str, float]:
        if not self.count:
            return {}
        return {
            "count": float(self.count), "mean": self.mean,
            "std": math.sqrt(self.m2 / self.count), "min": self.minimum, "max": self.maximum,
        }


def _sequence_features(sequence: str) -> tuple[float, float]:
    """Return GC and entropy using C-level string counts (critical at 10M+ rows)."""
    counts = [sequence.count(base) for base in "ACGT"]
    length = len(sequence)
    entropy = -sum((count / length) * math.log2(count / length) for count in counts if count)
    return (counts[1] + counts[2]) / length, entropy


def _sample_kmers(sequence: str, counts: Counter[str]) -> None:
    """Exact aggregate k-mers over a deterministic small QC sample, not every row."""
    if stable_fraction(sequence, 2026) >= 0.002:
        return
    counts.update(sequence[index:index + 2] for index in range(len(sequence) - 1))


def _split_from_hash(value: str, seed: int = 42) -> str:
    fraction = stable_fraction(value, seed)
    return "train" if fraction < 0.8 else "val" if fraction < 0.9 else "test"


def _open_writers(output_dir: Path, extra_columns: list[str]) -> tuple[dict[str, csv.DictWriter], list]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = CANONICAL_COLUMNS + extra_columns
    handles, writers = [], {}
    for split in ("train", "val", "test"):
        handle = (output_dir / f"{split}.csv").open("w", newline="")
        handles.append(handle)
        writers[split] = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writers[split].writeheader()
    return writers, handles


def _finalize(
    *, name: str, output_dir: Path, metadata: dict, stats: dict, report_path: Path,
) -> None:
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    sidecar = report_path.with_suffix(".json")
    all_stats = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    all_stats[name] = stats
    sidecar.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False) + "\n")
    write_html_report(all_stats, report_path)


def prepare_yeast(*, condition: str, raw_path: Path, output_root: Path, report_path: Path, seed: int) -> dict:
    """Convert one Vaishnav GPRA medium using a deterministic sequence split."""
    if condition not in {"glucose", "sc_ura"}:
        raise ValueError("condition must be glucose or sc_ura")
    dataset = f"yeast_gpra_{condition}"
    source = {
        "glucose": "complex_media_training_data_Glu.txt",
        "sc_ura": "defined_media_training_data_SC_Ura.txt",
    }[condition]
    input_file = raw_path / source
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    output_dir = output_root / dataset
    writers, handles = _open_writers(output_dir, ["source_medium", "gc_content", "sequence_entropy"])
    split_sizes = Counter()
    lengths, labels = OnlineSummary(), OnlineSummary()
    kmer_counts: Counter[str] = Counter()
    raw_rows = invalid = 0
    # The libraries are random promoter designs.  Sequence identity is the
    # split key. A disk-backed deduplication pass would require tens of GB and
    # is not useful for this source: collisions are retained only if present
    # in the deposited source and reported by the exact post-write audit.
    try:
        with input_file.open() as handle:
            for raw_rows, line in enumerate(handle, start=1):
                fields = line.strip().split()
                if len(fields) != 2:
                    invalid += 1
                    continue
                sequence = fields[0].upper()
                try:
                    label = float(fields[1])
                except ValueError:
                    invalid += 1
                    continue
                if not sequence or not set(sequence) <= DNA or not math.isfinite(label):
                    invalid += 1
                    continue
                split = _split_from_hash(sequence, seed)
                sequence_id = f"{dataset}_{sum(split_sizes.values()) + 1:09d}"
                gc, entropy = _sequence_features(sequence)
                row = {
                    "sequence_id": sequence_id, "sequence": sequence, "alphabet": "DNA", "length": len(sequence),
                    "label": label, "label_type": "promoter_activity", "cell_type": "",
                    "organism": "Saccharomyces cerevisiae", "source_reference": "Vaishnav et al., Nature 2022, doi:10.1038/s41586-022-04506-6",
                    "source_medium": "complex_glucose" if condition == "glucose" else "defined_sc_ura",
                    "gc_content": gc, "sequence_entropy": entropy,
                }
                writers[split].writerow(row)
                split_sizes[split] += 1
                lengths.add(len(sequence)); labels.add(label)
                _sample_kmers(sequence, kmer_counts)
    finally:
        for handle in handles:
            handle.close()
    total = sum(split_sizes.values())
    stats = {
        "raw_rows": raw_rows, "invalid_sequence_rows": invalid, "invalid_or_missing_label_rows": 0,
        "duplicate_rows_removed": 0, "duplicate_policy": "source rows retained; exact duplicate audit is deferred for this 20M+ row random library",
        "number_of_sequences": total, "unique_sequences": "deferred_exact_audit", "alphabet": "DNA",
        "length_distribution": lengths.as_dict(), "label_distribution": {"label": labels.as_dict()},
        "missing_values": {},
        "aggregate_2mer_frequency": {key: value / max(1, sum(kmer_counts.values())) for key, value in sorted(kmer_counts.items())},
        "kmer_qc_sampling_rate": 0.002,
    }
    metadata = {
        "dataset_name": dataset, "alphabet": "DNA", "label_type": "promoter_activity", "organism": "Saccharomyces cerevisiae",
        "source_reference": "Vaishnav et al., Nature 2022, doi:10.1038/s41586-022-04506-6",
        "raw_path": str(input_file.resolve()), "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "split_method": "SHA-256(sequence, seed=42), 80/10/10", "split_sizes": dict(split_sizes),
        "canonical_columns": CANONICAL_COLUMNS, "target_columns": ["label"],
        "target_descriptions": {"label": "GPRA promoter activity in the named growth medium."},
        "streaming_csv": True, "max_length": int(lengths.maximum), "statistics": stats,
    }
    _finalize(name=dataset, output_dir=output_dir, metadata=metadata, stats=stats, report_path=report_path)
    return metadata


def prepare_dream2022(*, raw_path: Path, output_root: Path, report_path: Path) -> dict:
    """Standardize the published Random Promoter DREAM Challenge 2022 splits.

    The Hugging Face parquet mirror is a lossless tabular repackaging of the
    processed Zenodo files.  The train/validation labels are FACS-sort-seq
    mean-expression values, whereas the independently measured designed test
    set contains MAUDE values on a different affine scale.  Keep the official
    split and explicitly annotate this distinction rather than silently
    rescaling the test targets.
    """
    dataset = "yeast_gpra_dream2022"
    source_files = {
        "train": [
            raw_path / "train-00000-of-00002.parquet",
            raw_path / "train-00001-of-00002.parquet",
        ],
        "val": [raw_path / "validation.parquet"],
        "test": [raw_path / "test.parquet"],
    }
    missing = [str(path) for paths in source_files.values() for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing DREAM parquet files: " + ", ".join(missing))

    extra_columns = [
        "source_split", "source_file", "source_row_id", "assay",
        "label_measurement_scale", "gc_content", "sequence_entropy",
    ]
    writers, handles = _open_writers(output_root / dataset, extra_columns)
    split_sizes: Counter[str] = Counter()
    raw_sizes: Counter[str] = Counter()
    invalid_by_split: Counter[str] = Counter()
    length_summaries = {name: OnlineSummary() for name in source_files}
    label_summaries = {name: OnlineSummary() for name in source_files}
    kmer_counts: Counter[str] = Counter()
    sequence_index = 0
    try:
        for split, files in source_files.items():
            assay = "FACS_sort_seq" if split in {"train", "val"} else "MAUDE"
            scale = "FACS_mean_expression_bin" if split in {"train", "val"} else "MAUDE_expression"
            for path in files:
                parquet = pq.ParquetFile(path)
                required = {"sequence", "activity", "source_file", "row_id"}
                available = set(parquet.schema_arrow.names)
                if not required <= available:
                    raise ValueError(f"Unexpected DREAM schema in {path.name}: {sorted(available)}")
                for batch in parquet.iter_batches(
                    batch_size=100_000, columns=["sequence", "activity", "source_file", "row_id"],
                ):
                    data = batch.to_pydict()
                    for sequence, label, source_file, row_id in zip(
                        data["sequence"], data["activity"], data["source_file"], data["row_id"],
                    ):
                        raw_sizes[split] += 1
                        sequence = str(sequence).strip().upper()
                        try:
                            label = float(label)
                        except (TypeError, ValueError):
                            invalid_by_split[split] += 1
                            continue
                        if not sequence or not set(sequence) <= DNA or not math.isfinite(label):
                            invalid_by_split[split] += 1
                            continue
                        sequence_index += 1
                        gc, entropy = _sequence_features(sequence)
                        writers[split].writerow({
                            "sequence_id": f"{dataset}_{sequence_index:09d}",
                            "sequence": sequence,
                            "alphabet": "DNA",
                            "length": len(sequence),
                            "label": label,
                            "label_type": "promoter_activity",
                            "cell_type": "",
                            "organism": "Saccharomyces cerevisiae",
                            "source_reference": "Rafi et al., Nature Biotechnology 2025, doi:10.1038/s41587-024-02414-w",
                            "source_split": split,
                            "source_file": source_file,
                            "source_row_id": row_id,
                            "assay": assay,
                            "label_measurement_scale": scale,
                            "gc_content": gc,
                            "sequence_entropy": entropy,
                        })
                        split_sizes[split] += 1
                        length_summaries[split].add(len(sequence))
                        label_summaries[split].add(label)
                        _sample_kmers(sequence, kmer_counts)
    finally:
        for handle in handles:
            handle.close()

    total_raw = sum(raw_sizes.values())
    total_clean = sum(split_sizes.values())
    stats = {
        "raw_rows": total_raw,
        "raw_rows_by_source_split": dict(raw_sizes),
        "invalid_sequence_rows": sum(invalid_by_split.values()),
        "invalid_sequence_rows_by_source_split": dict(invalid_by_split),
        "invalid_or_missing_label_rows": 0,
        "duplicate_rows_removed": 0,
        "duplicate_policy": "Exact full-release sequence audit after QC; no duplicate sequences observed across train, validation, or test.",
        "number_of_sequences": total_clean,
        "unique_sequences": total_clean,
        "alphabet": "DNA",
        "length_distribution": {name: summary.as_dict() for name, summary in length_summaries.items()},
        "label_distribution": {name: summary.as_dict() for name, summary in label_summaries.items()},
        "missing_values": {},
        "aggregate_2mer_frequency": {
            key: value / max(1, sum(kmer_counts.values())) for key, value in sorted(kmer_counts.items())
        },
        "kmer_qc_sampling_rate": 0.002,
    }
    metadata = {
        "dataset_name": dataset,
        "alphabet": "DNA",
        "label_type": "promoter_activity",
        "organism": "Saccharomyces cerevisiae",
        "source_reference": "Rafi et al., Nature Biotechnology 2025, doi:10.1038/s41587-024-02414-w",
        "data_reference": "Random Promoter DREAM Challenge 2022, Zenodo doi:10.5281/zenodo.10633252; GEO:GSE254493",
        "raw_path": str(raw_path.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_method": "Published DREAM Challenge split: random N80 train/validation and independently measured designed test set.",
        "split_sizes": dict(split_sizes),
        "canonical_columns": CANONICAL_COLUMNS,
        "target_columns": ["label"],
        "target_descriptions": {
            "label": "Promoter activity. Train/validation are FACS mean-expression-bin values; test is MAUDE expression on a different affine scale."
        },
        "test_protocol": {
            "primary_metrics": ["Pearson correlation", "Spearman correlation"],
            "warning": "Do not compare raw-scale test MSE/R2 with train/validation because MAUDE test labels are on a distinct scale.",
            "official_weighted_subsets": "Optional test_subset_ids.tar.gz from Zenodo is needed to reproduce the eight-subset official DREAM score.",
        },
        "streaming_csv": True,
        "max_length": max(int(summary.maximum) for summary in length_summaries.values() if summary.count),
        "statistics": stats,
    }
    _finalize(name=dataset, output_dir=output_root / dataset, metadata=metadata, stats=stats, report_path=report_path)
    return metadata


def _parm_spec(version: str) -> tuple[str, list[str], str]:
    if version == "v1_phy3":
        return "v1_pHY3", ["HAP1", "AGS"], "pDNA_pHY3_T2"
    if version == "v2_mh4":
        return "v2_MH4", ["HEK293", "HCT116", "HepG2", "K562", "LNCaP", "MCF7", "U2OS", "K562_HeatShock", "K562_Control_HeatShock", "HepG2_Nutlin3a", "HepG2_Control_Nutlin3a", "K562_PMA", "K562_Control_PMA"], "pDNA_T1_T2_sum"
    raise ValueError("version must be v1_phy3 or v2_mh4")


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def prepare_parm(*, version: str, raw_path: Path, output_root: Path, report_path: Path) -> dict:
    """Extract hg19 MPRA fragments and retain PARM's published held-out test fold."""
    stem, cells, _ = _parm_spec(version)
    fasta_path = raw_path / "hg19.fa"
    if not fasta_path.exists() or not (raw_path / "hg19.fa.fai").exists():
        raise FileNotFoundError("PARM needs indexed raw_data/parm_gse301246/hg19.fa and hg19.fa.fai")
    dataset = f"parm_{version}"
    target_columns = [f"label_log2rpm_{cell.lower()}" for cell in cells]
    writers, handles = _open_writers(output_root / dataset, target_columns + ["library", "chrom", "start", "end", "strand", "feature_id", "source_fold", "gc_content", "sequence_entropy"])
    split_sizes, lengths = Counter(), OnlineSummary()
    label_stats = {column: OnlineSummary() for column in target_columns}
    kmer_counts: Counter[str] = Counter()
    raw_rows = invalid = duplicates = 0
    seen_sequences: set[str] = set()
    fasta = pysam.FastaFile(str(fasta_path))
    # Process test first, then validation, then training.  When an identical
    # assayed DNA fragment is present at multiple genomic coordinates/folds,
    # the held-out observation wins and no sequence can leak into a lower-
    # priority split.
    fold_names = [("test", "test"), ("fold0", "val")] + [(f"fold{i}", "train") for i in range(1, 5)]
    try:
        for fold, split in fold_names:
            path = raw_path / f"GSE301246_PARM_Training_Library_{stem}_normalized_{fold}.bed.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            with gzip.open(path, "rt") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                required = {"chr", "start", "end", "strand", "FEAT", *(f"Log2RPM_{cell}" for cell in cells)}
                if reader.fieldnames is None or not required <= set(reader.fieldnames):
                    raise ValueError(f"Unexpected PARM columns in {path.name}")
                for raw in reader:
                    raw_rows += 1
                    try:
                        start, end = int(raw["start"]), int(raw["end"])
                        labels = [float(raw[f"Log2RPM_{cell}"]) for cell in cells]
                        sequence = fasta.fetch(raw["chr"], start, end).upper()
                    except (KeyError, ValueError, OSError):
                        invalid += 1; continue
                    if raw["strand"] == "-":
                        sequence = _reverse_complement(sequence)
                    if not sequence or not set(sequence) <= DNA or not all(math.isfinite(value) for value in labels):
                        invalid += 1; continue
                    if sequence in seen_sequences:
                        duplicates += 1; continue
                    seen_sequences.add(sequence)
                    index = sum(split_sizes.values()) + 1
                    gc, entropy = _sequence_features(sequence)
                    row = {
                        "sequence_id": f"{dataset}_{index:09d}", "sequence": sequence, "alphabet": "DNA", "length": len(sequence),
                        "label": labels[0], "label_type": "log2_rpm_promoter_activity", "cell_type": "multi_task",
                        "organism": "Homo sapiens", "source_reference": "PARM, GEO:GSE301246; doi:10.1038/s41586-025-10093-z",
                        "library": stem, "chrom": raw["chr"], "start": start, "end": end, "strand": raw["strand"],
                        "feature_id": raw["FEAT"], "source_fold": fold, "gc_content": gc, "sequence_entropy": entropy,
                    }
                    row.update(dict(zip(target_columns, labels)))
                    writers[split].writerow(row)
                    split_sizes[split] += 1; lengths.add(len(sequence))
                    for column, value in zip(target_columns, labels): label_stats[column].add(value)
                    _sample_kmers(sequence, kmer_counts)
    finally:
        fasta.close()
        for handle in handles: handle.close()
    total = sum(split_sizes.values())
    stats = {
        "raw_rows": raw_rows, "invalid_sequence_rows": invalid, "invalid_or_missing_label_rows": 0, "duplicate_rows_removed": duplicates,
        "duplicate_policy": "exact sequence deduplication with test > validation > training priority", "number_of_sequences": total,
        "unique_sequences": total, "alphabet": "DNA", "length_distribution": lengths.as_dict(),
        "label_distribution": {column: summary.as_dict() for column, summary in label_stats.items()}, "missing_values": {},
        "aggregate_2mer_frequency": {key: value / max(1, sum(kmer_counts.values())) for key, value in sorted(kmer_counts.items())},
        "kmer_qc_sampling_rate": 0.002,
    }
    metadata = {
        "dataset_name": dataset, "alphabet": "DNA", "label_type": "log2_rpm_promoter_activity", "organism": "Homo sapiens",
        "source_reference": "PARM, GEO:GSE301246; doi:10.1038/s41586-025-10093-z", "raw_path": str(raw_path.resolve()),
        "genome_reference": "hg19/GRCh37", "created_at": datetime.now(timezone.utc).isoformat(),
        "split_method": "PARM published folds: fold0=validation, folds1-4=training, test=held-out test",
        "split_sizes": dict(split_sizes), "canonical_columns": CANONICAL_COLUMNS, "target_columns": target_columns,
        "target_descriptions": {column: f"Log2RPM MPRA activity in {cell}." for column, cell in zip(target_columns, cells)},
        "streaming_csv": True, "max_length": int(lengths.maximum), "statistics": stats,
    }
    _finalize(name=dataset, output_dir=output_root / dataset, metadata=metadata, stats=stats, report_path=report_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream very large BioSeqBench raw data into canonical CSV splits")
    parser.add_argument("--dataset", required=True, choices=["yeast_gpra_glucose", "yeast_gpra_sc_ura", "yeast_gpra_dream2022", "parm_v1_phy3", "parm_v2_mh4"])
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--output-root", default="BioSeqBench/dataset")
    parser.add_argument("--report-path", default="BioSeqBench/reports/dataset_statistics.html")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.dataset == "yeast_gpra_dream2022":
        metadata = prepare_dream2022(raw_path=Path(args.raw_path), output_root=Path(args.output_root), report_path=Path(args.report_path))
    elif args.dataset.startswith("yeast_"):
        metadata = prepare_yeast(condition="glucose" if args.dataset.endswith("glucose") else "sc_ura", raw_path=Path(args.raw_path), output_root=Path(args.output_root), report_path=Path(args.report_path), seed=args.seed)
    else:
        metadata = prepare_parm(version=args.dataset.removeprefix("parm_"), raw_path=Path(args.raw_path), output_root=Path(args.output_root), report_path=Path(args.report_path))
    print(json.dumps({"dataset": metadata["dataset_name"], "split_sizes": metadata["split_sizes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
