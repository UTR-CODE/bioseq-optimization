"""Zhou et al. (2026) L3 E. coli -10/-35 promoter library loader."""

from pathlib import Path

import pandas as pd

from .base import DatasetLoader


class Zhou2026PromoterLoader(DatasetLoader):
    name = "Zhou2026_L3_promoters"
    alphabet = "DNA"
    label_type = "relative_promoter_strength"
    organism = "Escherichia coli"
    source_reference = "Zhou et al., ACS Synthetic Biology 2026, doi:10.1021/acssynbio.5c00765"
    target_descriptions = {"label": "Processed L3 promoter strength from the authors' FACS-seq analysis."}
    coverage_check = {"published_L3_promoters": 20799, "library": "L3"}
    metadata_extras = {
        "task_type": "conditional_sequence_regression",
        "context_column": "cell_type",
        "context_values": ["L3_s1", "L3_s2", "L3_s3", "L3_s4"],
        "context_note": "These values encode the four documented L3 source sublibraries, not biological cell types."
    }

    def load(self, raw_path: Path) -> pd.DataFrame:
        path = Path(raw_path) / "sb5c00765_si_002.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"Zhou 2026 workbook not found: {path}")
        frame = pd.read_excel(path, sheet_name="Sequence and strength of L3")
        required = {"Sequences", "Strength"}
        if not required <= set(frame):
            raise ValueError("Unexpected Zhou 2026 L3 worksheet columns")
        # Table S2's aggregate L3 table intentionally omits the four source
        # sublibrary identifiers.  They are recoverable without inference from
        # the accompanying s1--s4 worksheets: every listed sequence occurs
        # once and carries exactly its aggregate-table strength.  This context
        # is essential because the four experimental sublibraries have markedly
        # different response distributions and their membership is not reliably
        # inferable from sequence alone.
        contexts = []
        for sublibrary in ("s1", "s2", "s3", "s4"):
            subset = pd.read_excel(path, sheet_name=sublibrary)
            if not required <= set(subset):
                raise ValueError(f"Unexpected Zhou 2026 {sublibrary} worksheet columns")
            subset = subset[["Sequences", "Strength"]].copy()
            subset["sublibrary"] = sublibrary
            contexts.append(subset)
        context = pd.concat(contexts, ignore_index=True)
        context = context.rename(columns={"Sequences": "sequence", "Strength": "source_strength"})
        aggregate = frame[["Sequences", "Strength"]].rename(columns={"Sequences": "sequence", "Strength": "label"})
        merged = aggregate.merge(context, on="sequence", how="inner", validate="one_to_one")
        if len(merged) != len(aggregate) or len(merged) != 20799:
            raise ValueError(f"Expected exactly 20,799 L3 sequence/context records; observed {len(merged)}")
        if not pd.to_numeric(merged["label"], errors="coerce").equals(pd.to_numeric(merged["source_strength"], errors="coerce")):
            raise ValueError("L3 aggregate strengths do not agree with source sublibrary worksheets")
        return pd.DataFrame({
            "sequence": merged["sequence"], "label": pd.to_numeric(merged["label"], errors="coerce"),
            # Reuse the canonical context field so SequenceDataset can provide
            # it as a categorical conditional-model input.
            "cell_type": "L3_" + merged["sublibrary"].astype(str),
            "sublibrary": merged["sublibrary"], "assay": "FACS_seq_processed_promoter_strength",
        })
