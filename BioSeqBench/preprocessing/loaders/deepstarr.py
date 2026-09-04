from pathlib import Path

import pandas as pd

from .base import DatasetLoader, pick_column, read_table


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    ids, sequences = [], []
    current_id, parts = None, []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    ids.append(current_id)
                    sequences.append("".join(parts))
                current_id, parts = line[1:].split()[0], []
            elif line:
                parts.append(line)
    if current_id is not None:
        ids.append(current_id)
        sequences.append("".join(parts))
    return ids, sequences


def reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(str.maketrans("ACGT", "TGCA"))[::-1]


def remove_rc_augmentation(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the appended reverse-complement half of an augmented split.

    Some processed DeepSTARR mirrors distribute each publication split followed
    by its reverse-complement augmentation.  These are not independent MPRA
    measurements and must not inflate a sequence-optimization benchmark.
    The source IDs and several positions are checked before retaining the
    original, forward-orientation first half.
    """
    if len(frame) % 2 or "sequence" not in frame:
        return frame
    half = len(frame) // 2
    probes = sorted({0, half // 2, half - 1})
    if not all(reverse_complement(frame["sequence"].iloc[i]) == frame["sequence"].iloc[i + half] for i in probes):
        return frame
    return frame.iloc[:half].copy()


class DeepSTARRLoader(DatasetLoader):
    name = "DeepSTARR"
    alphabet = "DNA"
    label_type = "enhancer_activity"
    organism = "Drosophila melanogaster (S2)"
    source_reference = "de Almeida et al. Nat Genet 2022; doi:10.1038/s41588-022-01048-5; Zenodo:5502060"

    def load(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.is_dir():
            frames = []
            for filename, split in (("train.parquet", "train"), ("valid.parquet", "val"), ("test.parquet", "test")):
                parquet = raw_path / filename
                if parquet.exists():
                    part = pd.read_parquet(parquet)
                    part = remove_rc_augmentation(part)
                    part["source_split"] = split
                    if "id" in part:
                        part["source_sequence_id"] = part["id"]
                    frames.append(part)
            for split in ("Train", "Val", "Test"):
                normalized_split = split.lower().replace("val", "val")
                if any(existing["source_split"].iloc[0] == normalized_split for existing in frames):
                    continue
                fasta = raw_path / f"Sequences_{split}.fa"
                labels = raw_path / f"Sequences_activity_{split}.txt"
                if fasta.exists() and labels.exists():
                    ids, sequences = read_fasta(fasta)
                    activity = pd.read_csv(labels, sep="\t")
                    if len(activity) != len(sequences):
                        raise ValueError(f"DeepSTARR {split} FASTA/activity row count mismatch")
                    activity.insert(0, "sequence", sequences)
                    activity["source_sequence_id"] = ids
                    activity["source_split"] = normalized_split
                    frames.append(activity)
            if frames:
                frame = pd.concat(frames, ignore_index=True)
            else:
                fasta = next(iter(sorted(raw_path.glob("*All.fa"))), None)
                labels = next(iter(sorted(raw_path.glob("*activity_All.txt"))), None)
                if not fasta or not labels:
                    raise FileNotFoundError("DeepSTARR directory needs matching Train/Val/Test or All FASTA/activity files")
                ids, sequences = read_fasta(fasta)
                activity = pd.read_csv(labels, sep="\t")
                if len(activity) != len(sequences):
                    raise ValueError("DeepSTARR FASTA/activity row count mismatch")
                activity.insert(0, "sequence", sequences)
                activity["source_sequence_id"] = ids
                frame = activity
        else:
            frame = read_table(raw_path, sep="\t")
        seq = pick_column(frame, ("sequence", "seq"))
        dev = pick_column(frame, ("dev", "developmental", "developmental_activity", "dev_log2_enrichment"))
        hk = pick_column(frame, ("hk", "housekeeping", "housekeeping_activity", "hk_log2_enrichment"))
        output = pd.DataFrame({
            "sequence": frame[seq], "label": frame[dev],
            "label_developmental": frame[dev], "label_housekeeping": frame[hk],
        })
        if "source_sequence_id" in frame:
            output["source_sequence_id"] = frame["source_sequence_id"]
        if "source_split" in frame:
            output["source_split"] = frame["source_split"]
        return output
