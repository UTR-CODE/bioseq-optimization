"""Agarwal et al. (2025) lentiMPRA library loaders."""

from pathlib import Path

import pandas as pd

from .base import DatasetLoader


class Agarwal2025LentiMPRALoader(DatasetLoader):
    """Load the fold-assigned, three-cell-type joint lentiMPRA library.

    The design sheet stores a 200-nt assayed element flanked by fixed 15-nt
    cloning adaptors.  Only the central element is exposed to the model.
    """

    name = "Agarwal2025_lentiMPRA"
    alphabet = "DNA"
    label_type = "log2_rna_dna_activity"
    organism = "Homo sapiens"
    source_reference = "Agarwal et al., Nature 2025, doi:10.1038/s41586-024-08430-9"
    target_descriptions = {
        "label": "HepG2 log2(RNA/DNA), primary target.",
        "label_hepg2": "HepG2 log2(RNA/DNA) lentiMPRA activity.",
        "label_k562": "K562 log2(RNA/DNA) lentiMPRA activity.",
        "label_wtc11": "WTC11 log2(RNA/DNA) lentiMPRA activity.",
    }
    coverage_check = {
        "joint_library_designs": 60000,
        "joint_activity_table_rows": 56981,
        "design_activity_id_overlap_rows": 56979,
        "official_fold_assigned_rows": 55338,
        "element_length_nt": 200,
        "adaptor_length_each_side_nt": 15,
    }
    metadata_extras = {
        "split_method": "publication 10-fold CV assignment; benchmark convention: fold 1=test, fold 2=validation, folds 3-10=training",
        "benchmark_split_note": "The publication evaluates all ten folds. The single train/val/test partition is a reproducible BioSeqBench convention, not a unique paper-held-out test set.",
    }

    def load(self, raw_path: Path) -> pd.DataFrame:
        root = Path(raw_path)
        source = root / "2023-03-03628-s5"
        if not source.exists():
            source = root
        design_file = source / "2023-03-03628C-Table S10 - joint lib design.xlsx"
        activity_file = source / "2023-03-03628C-Table S11 - jointdata.xlsx"
        fold_file = source / "2023-03-03628C-Table S12 - folds and performance.xlsx"
        missing = [str(path) for path in (design_file, activity_file, fold_file) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Agarwal joint-library source files missing: {missing}")
        design = pd.read_excel(design_file, sheet_name="joint library", header=1)
        activity = pd.read_excel(activity_file, sheet_name="all_cell_types_summary")
        folds = pd.read_excel(fold_file, sheet_name="Fold assignment, joint library")
        sequence_column = "230nt sequence (15nt 5' adaptor - 200nt element - 15nt 3' adaptor)"
        labels = {
            "HepG2 [log2(rna/dna)]": "label_hepg2",
            "K562 [log2(rna/dna)]": "label_k562",
            "WTC11 [log2(rna/dna)]": "label_wtc11",
        }
        required_design = {"name", "category", sequence_column}
        if not required_design <= set(design) or not {"name", *labels} <= set(activity) or not {"Fold", "Sequence ID"} <= set(folds):
            raise ValueError("Unexpected Agarwal 2025 joint-library worksheet columns")
        joined = design.merge(activity, on="name", how="inner", validate="one_to_one")
        joined = joined.merge(folds[["Fold", "Sequence ID"]], left_on="name", right_on="Sequence ID", how="inner", validate="one_to_one")
        sequence_230 = joined[sequence_column].astype(str).str.upper()
        if not sequence_230.str.len().eq(230).all():
            raise ValueError("Expected all Agarwal joint-library oligos to be 230 nt")
        # Give a concrete benchmark train/validation/test partition while
        # retaining the publication's full 10-fold identifier for CV reuse.
        source_split = joined["Fold"].map(lambda fold: "test" if fold == 1 else "val" if fold == 2 else "train")
        output = pd.DataFrame({
            "sequence": sequence_230.str.slice(15, -15),
            "label": pd.to_numeric(joined["HepG2 [log2(rna/dna)]"], errors="coerce"),
            "label_hepg2": pd.to_numeric(joined["HepG2 [log2(rna/dna)]"], errors="coerce"),
            "label_k562": pd.to_numeric(joined["K562 [log2(rna/dna)]"], errors="coerce"),
            "label_wtc11": pd.to_numeric(joined["WTC11 [log2(rna/dna)]"], errors="coerce"),
            "cell_type": "multi_task",
            "element_id": joined["name"].astype(str),
            "element_category": joined["category"].astype(str),
            "publication_fold": joined["Fold"].astype(int),
            "source_split": source_split,
            "split_group": joined["name"].astype(str),
            "assay": "lentiMPRA_joint_library",
        })
        # The author-provided fold sheet has 55,338 element IDs.  These are the
        # rows with a published ten-fold assignment after joining the 60k design
        # and joint-activity tables; all of them have complete three-cell labels
        # in the downloaded supplementary tables.
        return output.dropna(subset=["label_hepg2", "label_k562", "label_wtc11"]).reset_index(drop=True)


class _Agarwal2025LargeScaleLoader(DatasetLoader):
    """One of the three author-released MPRALegNet training tables.

    The raw files are the exact ``datasets/original/<cell>.tsv`` tables
    published with ``autosome-ru/human_legnet``.  Those files contain a
    ``rev`` column with a full-construct reverse copy; the author
    ``SeqDataModule`` explicitly filters to ``rev == 0`` before every model
    fit, then adds random RC augmentation during training.  We apply that same
    filter here.  A retained row's 230-nt sequence includes the 15-nt cloning
    adapters, as required by the MPRALegNet architecture.
    """

    cell_type: str
    table_name: str
    name: str
    alphabet = "DNA"
    label_type = "log2_rna_dna_activity"
    organism = "Homo sapiens"
    source_reference = "Agarwal et al., Nature 2025, doi:10.1038/s41586-024-08430-9"
    target_descriptions = {"label": "Large-scale lentiMPRA log2(RNA/DNA) activity."}
    metadata_extras = {
        "split_method": "Publication ten-fold CV: fold 1=test, fold 2=validation, folds 3-10=training for the canonical BioSeqBench view.",
        "benchmark_split_note": "MPRALegNet should use all ten folds for source-protocol CV; the canonical train/val/test view is only a reproducible convenience split.",
        "sequence_context": "230-nt construct: 15-nt 5-prime adapter + 200-nt tested element + 15-nt 3-prime adapter; only author rev=0 rows are retained, matching human_legnet datamodule.py.",
    }

    def load(self, raw_path: Path) -> pd.DataFrame:
        root = Path(raw_path)
        table = root / "human_legnet" / self.table_name
        if not table.exists():
            raise FileNotFoundError(
                f"Missing author MPRALegNet training table: {table}. "
                "Download datasets/original/{HepG2,K562,WTC11}.tsv from "
                "https://github.com/autosome-ru/human_legnet into raw_data/agarwal2025_lentimpra/human_legnet/."
            )
        frame = pd.read_csv(table, sep="\t")
        required = {"seq_id", "seq", "mean_value", "fold"}
        if not required <= set(frame):
            raise ValueError(f"Unexpected columns in {table}: {list(frame.columns)}")
        if "rev" in frame:
            reverse_copy = pd.to_numeric(frame["rev"], errors="raise").astype(int)
            if not reverse_copy.isin((0, 1)).all():
                raise ValueError(f"{table} rev column must contain only 0/1")
            frame = frame.loc[reverse_copy.eq(0)].copy()
        sequence = frame["seq"].astype(str).str.upper()
        if not sequence.str.len().eq(230).all():
            raise ValueError(f"{table} must contain fixed 230-nt constructs")
        folds = pd.to_numeric(frame["fold"], errors="raise").astype(int)
        if not folds.isin(range(1, 11)).all():
            raise ValueError(f"{table} contains folds outside 1..10")
        source_split = folds.map(lambda value: "test" if value == 1 else "val" if value == 2 else "train")
        return pd.DataFrame({
            "sequence": sequence,
            "label": pd.to_numeric(frame["mean_value"], errors="coerce"),
            "cell_type": self.cell_type,
            "element_id": frame["seq_id"].astype(str),
            "publication_fold": folds,
            "source_orientation": 0,
            "source_split": source_split,
            "split_group": frame["seq_id"].astype(str),
            "assay": "lentiMPRA_large_scale",
        })


class Agarwal2025LentiMPRAHepG2Loader(_Agarwal2025LargeScaleLoader):
    name = "agarwal2025_lentimpra_hepg2"
    cell_type = "HepG2"
    table_name = "HepG2.tsv"


class Agarwal2025LentiMPRAK562Loader(_Agarwal2025LargeScaleLoader):
    name = "agarwal2025_lentimpra_k562"
    cell_type = "K562"
    table_name = "K562.tsv"


class Agarwal2025LentiMPRAWTC11Loader(_Agarwal2025LargeScaleLoader):
    name = "agarwal2025_lentimpra_wtc11"
    cell_type = "WTC11"
    table_name = "WTC11.tsv"
