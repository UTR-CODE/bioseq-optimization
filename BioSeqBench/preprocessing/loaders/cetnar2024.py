"""Cetnar et al. (2024) bacterial mRNA-stability MPRA loader."""

import io
import zipfile
from pathlib import Path

import pandas as pd

from .base import DatasetLoader


class Cetnar2024MRNAStabilityLoader(DatasetLoader):
    """Load the authors' released feature-complete modelling split.

    ``SupplementaryData1.xlsx`` contains all 62,120 designs, whereas the
    authors' LightGBM training release contains a fixed 36,219/9,064 split.
    The release script retrieves the target ``k_fit`` from the workbook by the
    1-based ``Index``; it does *not* use the feature-table ``Kdeg`` field as
    the training label.  We preserve that exact split and retain positive
    ``k_fit`` observations only, as requested for the sequence benchmark.
    """

    name = "Cetnar2024_mRNA_stability"
    alphabet = "RNA"
    label_type = "mrna_decay_rate_per_min"
    organism = "Escherichia coli"
    source_reference = (
        "Cetnar et al., Nature Communications 2024, "
        "doi:10.1038/s41467-024-54059-7"
    )
    target_descriptions = {
        "label": "Fitted first-order mRNA degradation rate k_fit (min^-1); lower is more stable.",
        "half_life_min": "Derived half-life in minutes (ln(2) / k_fit), retained as auxiliary metadata.",
    }
    coverage_check = {
        "source_designs": 62120,
        "source_design_type": "barcoded synthetic 5-prime UTR variants",
        "paper_feature_table_rows": 45283,
        "paper_feature_table_positive_k_fit_rows": 45180,
        "excluded_positive_rows_not_in_paper_feature_tables": 12600,
        "excluded_nonpositive_k_fit_rows_from_full_workbook": 4340,
    }
    metadata_extras = {
        "paper_model_comparison": {
            "paper_repository": "https://github.com/hsalis/SalisLabCode/tree/master/mRNA_Stability_Calculator",
            "paper_feature_tables": {
                "training_rows_before_positive_k_fit_filter": 36219,
                "test_rows_before_positive_k_fit_filter": 9064,
                "training_rows": 36138,
                "test_rows": 9042,
                "total_rows": 45180,
                "label_lookup": "ModelTrainingTesting.py reads k_fit from SupplementaryData1.xlsx by 1-based Index; its precomputed Kdeg column is not used as y."
            },
            "benchmark_difference": "Uses the authors' feature-complete train/test membership, filters nonpositive k_fit, and preserves independent source observations (including repeated sequences with distinct barcode-derived measurements)."
        }
    }

    @staticmethod
    def _read_feature_table(archive_path: Path, member: str) -> pd.DataFrame:
        if not archive_path.exists():
            raise FileNotFoundError(f"Cetnar 2024 author feature table not found: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            try:
                return pd.read_csv(io.BytesIO(archive.read(member)))
            except KeyError as exc:
                raise ValueError(f"Expected {member} inside {archive_path}") from exc

    def load(self, raw_path: Path) -> pd.DataFrame:
        path = Path(raw_path)
        if not path.is_dir():
            raise ValueError("Cetnar 2024 loader requires the raw-data directory containing workbook and author train/test ZIPs")
        workbook = path / "SupplementaryData1.xlsx"
        if not workbook.exists():
            raise FileNotFoundError(f"Cetnar 2024 workbook not found: {workbook}")

        frame = pd.read_excel(workbook, sheet_name="Sequences and Data")
        sequence_column = "Designed 5' UTR Sequence"
        required = {sequence_column, "k_fit", "half-life", "k_r2_fit"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Unexpected Cetnar workbook columns; missing {sorted(missing)}")

        train = self._read_feature_table(path / "TrainingDataset.csv.zip", "TrainingDataset.csv")
        test = self._read_feature_table(path / "TestDataset.csv.zip", "TestDataset.csv")
        required_feature_columns = {"Index", "UTR_Designed_Sequence"}
        for split_name, feature_table in (("train", train), ("test", test)):
            missing = required_feature_columns - set(feature_table.columns)
            if missing:
                raise ValueError(f"Unexpected Cetnar {split_name} feature-table columns; missing {sorted(missing)}")
            feature_table["source_split"] = split_name
        features = pd.concat([train, test], ignore_index=True)
        indices = pd.to_numeric(features["Index"], errors="raise").astype(int)
        if indices.duplicated().any() or indices.min() < 1 or indices.max() > len(frame):
            raise ValueError("Cetnar author feature-table Index values are not unique valid 1-based workbook indices")

        # The oligo library is stored as DNA.  The assayed reporter is RNA, so
        # use U in the benchmark sequence while retaining the source sequence
        # in raw_data unchanged.  Validate the feature-table sequence against
        # the indexed source row before taking its experimental labels.
        source = frame.iloc[indices.to_numpy() - 1].reset_index(drop=True)
        source_sequence = source[sequence_column].astype(str).str.upper().str.replace("T", "U", regex=False)
        feature_sequence = features["UTR_Designed_Sequence"].astype(str).str.upper().str.replace("T", "U", regex=False)
        if not source_sequence.equals(feature_sequence):
            raise ValueError("Cetnar feature-table sequences do not match SupplementaryData1 at their Index values")
        output = pd.DataFrame({
            "sequence": source_sequence,
            "label": pd.to_numeric(source["k_fit"], errors="coerce"),
            "half_life_min": pd.to_numeric(source["half-life"], errors="coerce"),
            "fit_r2": pd.to_numeric(source["k_r2_fit"], errors="coerce"),
            "rna_t0_counts": pd.to_numeric(source["RNA_T0"], errors="coerce"),
            "rna_t2_counts": pd.to_numeric(source["RNA_T2"], errors="coerce"),
            "rna_t4_counts": pd.to_numeric(source["RNA_T4"], errors="coerce"),
            "rna_t8_counts": pd.to_numeric(source["RNA_T8"], errors="coerce"),
            "rna_t16_counts": pd.to_numeric(source["RNA_T16"], errors="coerce"),
            "source_observation_id": "cetnar_index_" + indices.astype(str),
            "source_split": features["source_split"].to_numpy(),
            "assay": "rifampicin_transcriptional_shutoff_kinetic_mpra",
        })
        return output.loc[output["label"] > 0].reset_index(drop=True)
