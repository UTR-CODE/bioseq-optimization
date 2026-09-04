from pathlib import Path

import pandas as pd

from .base import DatasetLoader, pick_column, read_table


class IGVFLoader(DatasetLoader):
    name = "IGVF_cCRE_MPRA"
    alphabet = "DNA"
    label_type = "regulatory_activity"
    organism = "Homo sapiens"
    source_reference = "IGVF Data Portal MPRA datasets; https://data.igvf.org"

    def load(self, raw_path: Path) -> pd.DataFrame:
        frame = read_table(raw_path)
        seq = pick_column(frame, ("sequence", "oligo_sequence", "element_sequence", "seq"))
        label = pick_column(frame, ("activity", "activity_score", "log2fc", "log2_fold_change", "label"))
        cell = pick_column(frame, ("cell_type", "biosample", "sample", "cell_line"))
        output = pd.DataFrame({"sequence": frame[seq], "label": frame[label], "cell_type": frame[cell]})
        for candidate in ("element_id", "variant", "strand"):
            column = pick_column(frame, (candidate,), required=False)
            if column:
                output[candidate] = frame[column]
        return output

