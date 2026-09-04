"""Zhao et al. (2022) Ptrc synthetic-promoter library loader."""

from pathlib import Path

import pandas as pd

from .base import DatasetLoader


class Zhao2022PtrcLoader(DatasetLoader):
    name = "Zhao2022_Ptrc_promoters"
    alphabet = "DNA"
    label_type = "fluorescence_per_od600"
    organism = "Escherichia coli"
    source_reference = "Zhao et al., ACS Synthetic Biology 2022, doi:10.1021/acssynbio.1c00117"
    target_descriptions = {"label": "Promoter strength measured as fluorescence / OD600."}
    coverage_check = {"published_synthetic_promoters": 3665, "join_rule": "measurement ML<n> joins sequence PL<n>"}

    def load(self, raw_path: Path) -> pd.DataFrame:
        raw_path = Path(raw_path)
        measurement_file = raw_path / "sb1c00117_si_002.xlsx"
        sequence_file = raw_path / "sb1c00117_si_003.xlsx"
        if not measurement_file.exists() or not sequence_file.exists():
            raise FileNotFoundError("Zhao 2022 requires sb1c00117_si_002.xlsx and sb1c00117_si_003.xlsx")
        measured = pd.read_excel(measurement_file, sheet_name="3665 recombinant strains")
        sequences = pd.read_excel(sequence_file, sheet_name="3665 synthetic promoters")
        required_measured = {"Name", "Fluorescence/OD600"}
        required_sequence = {"Name", "Sequence"}
        if not required_measured <= set(measured) or not required_sequence <= set(sequences):
            raise ValueError("Unexpected Zhao 2022 supplementary workbook columns")
        measured = measured.rename(columns={"Name": "measurement_id", "Fluorescence/OD600": "label", "Error": "measurement_error"})
        # The measurement workbook calls constructs ML<n>, while the sequence
        # workbook calls the identical constructs PL<n>.
        measured["sequence_id_source"] = measured["measurement_id"].astype(str).str.replace(r"^ML", "PL", regex=True)
        sequences = sequences.rename(columns={"Name": "sequence_id_source", "Sequence": "sequence"})
        joined = measured.merge(sequences, on="sequence_id_source", how="inner", validate="one_to_one")
        if len(joined) != 3665:
            raise ValueError(f"Expected 3,665 matched Zhao promoters; observed {len(joined)}")
        return pd.DataFrame({
            "sequence": joined["sequence"], "label": pd.to_numeric(joined["label"], errors="coerce"),
            "source_promoter_id": joined["sequence_id_source"], "measurement_id": joined["measurement_id"],
            "measurement_error": pd.to_numeric(joined["measurement_error"], errors="coerce"),
            "assay": "FACS_fluorescence_per_OD600",
        })
