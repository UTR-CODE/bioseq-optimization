"""Kosuri et al. (2013) E. coli promoter--RBS composability loader."""

from pathlib import Path

import pandas as pd

from .base import DatasetLoader


def _clean_source_sequence(value: object) -> str:
    """Remove the source workbook's quotation marks and visual spacer."""
    return str(value).replace('"', '').replace("'", "").replace(" ", "").strip().upper()


class Kosuri2013ComposabilityLoader(DatasetLoader):
    """Build assayed promoter--RBS sequences from the three source workbooks.

    Dataset S3 reports an individual construct's RNA and protein measurements,
    while Dataset S1/S2 supply the actual promoter and RBS oligo sequences.
    We retain the 11,918 constructs passing the authors' RNA, protein, and DNA
    quality flags, and use protein expression as the primary scalar target.
    """

    name = "Kosuri2013_composability"
    alphabet = "DNA"
    label_type = "protein_expression"
    organism = "Escherichia coli K-12 MG1655"
    source_reference = "Kosuri et al., PNAS 2013, doi:10.1073/pnas.1301301110"
    target_descriptions = {
        "label": "Protein expression (primary reporter measurement; arbitrary fluorescence units).",
        "label_protein_expression": "Protein expression (arbitrary fluorescence units).",
        "label_rna_expression": "RNA abundance (RNA/DNA-normalized measurement).",
    }
    coverage_check = {
        "source_construct_rows": 12655,
        "source_promoters": 112,
        "source_rbs_sequences": 111,
        "retained_policy": "finite RNA and protein measurement; bad.RNA, bad.prot, and bad.DNA all false",
    }

    def load(self, raw_path: Path) -> pd.DataFrame:
        raw_path = Path(raw_path)
        if not raw_path.is_dir():
            raise ValueError("Kosuri loader expects a directory containing sd01.xls, sd02.xls, and sd03.xls")
        files = {name: raw_path / name for name in ("sd01.xls", "sd02.xls", "sd03.xls")}
        missing = [str(path) for path in files.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing Kosuri supplementary workbooks: {missing}")

        promoters = pd.read_excel(files["sd01.xls"], sheet_name="Promoters")
        rbss = pd.read_excel(files["sd02.xls"], sheet_name="RBSs")
        constructs = pd.read_excel(files["sd03.xls"], sheet_name="Constructs")
        for frame, identifier in ((promoters, "Promoter"), (rbss, "RBS")):
            if identifier not in frame or "Sequence" not in frame:
                raise ValueError(f"Unexpected Kosuri sequence workbook: need {identifier!r} and 'Sequence'")

        promoter_sequences = dict(zip(promoters["Promoter"], promoters["Sequence"].map(_clean_source_sequence)))
        rbs_sequences = dict(zip(rbss["RBS"], rbss["Sequence"].map(_clean_source_sequence)))
        required = {"Promoter", "RBS", "target", "RNA", "prot", "bad.RNA", "bad.prot", "bad.DNA"}
        missing_columns = required - set(constructs.columns)
        if missing_columns:
            raise ValueError(f"Unexpected Kosuri construct workbook; missing {sorted(missing_columns)}")

        frame = constructs.copy()
        frame["promoter_sequence"] = frame["Promoter"].map(promoter_sequences)
        frame["rbs_sequence"] = frame["RBS"].map(rbs_sequences)
        pass_source_qc = (
            frame["RNA"].notna()
            & frame["prot"].notna()
            & ~frame["bad.RNA"].fillna(True).astype(bool)
            & ~frame["bad.prot"].fillna(True).astype(bool)
            & ~frame["bad.DNA"].fillna(True).astype(bool)
            & frame["promoter_sequence"].notna()
            & frame["rbs_sequence"].notna()
        )
        frame = frame.loc[pass_source_qc].copy()
        # The assay constructs concatenate the independently supplied promoter
        # and RBS variable oligos.  The adjacent flanking bases in Dataset S1/S2
        # are deliberately retained because they are part of the assayed DNA.
        output = pd.DataFrame({
            "sequence": frame["promoter_sequence"] + frame["rbs_sequence"],
            "label": pd.to_numeric(frame["prot"], errors="coerce"),
            "label_protein_expression": pd.to_numeric(frame["prot"], errors="coerce"),
            "label_rna_expression": pd.to_numeric(frame["RNA"], errors="coerce"),
            "promoter_id": frame["Promoter"].astype(str),
            "rbs_id": frame["RBS"].astype(str),
            "source_construct_id": frame["target"].astype(str),
            "promoter_sequence": frame["promoter_sequence"],
            "rbs_sequence": frame["rbs_sequence"],
            "translation_delta_g": pd.to_numeric(frame["deltaG"], errors="coerce"),
            "source_qc_pass": True,
            "assay": "combinatorial_promoter_rbs_reporter",
            "split_group": frame["target"].astype(str),
        })
        return output.reset_index(drop=True)
